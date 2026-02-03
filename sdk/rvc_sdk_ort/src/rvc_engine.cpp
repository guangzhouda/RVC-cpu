// rvc_engine.cpp

#include "rvc_engine.h"

#include <algorithm>
#include <cstdlib>
#include <cmath>
#include <cstring>
#ifdef _WIN32
#include <Windows.h>
#endif

#include "dsp/linear_resampler.h"
#include "dsp/sola.h"
#include "f0/rmvpe_f0.h"
#include "f0/yin_f0.h"

// ORT
#include <onnxruntime_cxx_api.h>
#include <onnxruntime_c_api.h>

// FAISS
#include <faiss/Index.h>
#include <faiss/index_io.h>

namespace rvc_ort {

static void SetError(Error* err, int32_t code, const std::string& msg) {
  if (!err) return;
  err->code = code;
  err->message = msg;
}

#ifdef _WIN32
static void* GetOrtExportOrNull_(const char* export_name) {
  if (!export_name || export_name[0] == '\0') return nullptr;
  // 说明：rvc_sdk_ort.dll 通过 import table 依赖 onnxruntime.dll，正常情况下这里一定能拿到句柄。
  // 但为了稳妥（例如宿主延迟加载/自定义加载），这里允许回退到 LoadLibrary。
  HMODULE mod = GetModuleHandleW(L"onnxruntime.dll");
  if (!mod) {
    mod = LoadLibraryW(L"onnxruntime.dll");
  }
  if (!mod) return nullptr;
  return reinterpret_cast<void*>(GetProcAddress(mod, export_name));
}
#endif

static float Pow2(float x) { return x * x; }

static float SemitoneToRatio(int32_t semitone) {
  return std::pow(2.0f, static_cast<float>(semitone) / 12.0f);
}

static void BuildFadeWindows(int32_t n, std::vector<float>* fade_in, std::vector<float>* fade_out) {
  fade_in->assign(std::max<int32_t>(0, n), 0.0f);
  fade_out->assign(std::max<int32_t>(0, n), 0.0f);
  if (n <= 0) return;
  if (n == 1) {
    (*fade_in)[0] = 1.0f;
    (*fade_out)[0] = 0.0f;
    return;
  }
  // 与 python 对齐：sin^2 窗
  constexpr float kPi = 3.14159265358979323846f;
  for (int32_t i = 0; i < n; ++i) {
    const float x = static_cast<float>(i) / static_cast<float>(n - 1);
    const float s = std::sin(0.5f * kPi * x);
    const float w = s * s;
    (*fade_in)[i] = w;
    (*fade_out)[i] = 1.0f - w;
  }
}

static bool CheckDivisibleBy100(int32_t sr) {
  return sr > 0 && (sr % 100) == 0;
}

#ifdef _WIN32
static std::wstring Utf8OrAcpToWide_(const std::string& s) {
  if (s.empty()) return std::wstring();

  // 说明：Windows 控制台/宿主程序传入的 char* 往往是系统 ANSI 代码页（如中文环境的 CP936），
  // 但也可能是 UTF-8（例如 `chcp 65001` 或宿主以 UTF-8 构造参数）。
  // 这里先尝试 UTF-8，失败则回退到 CP_ACP，尽量做到“拿到什么编码都能跑”。
  auto convert = [&](UINT cp) -> std::wstring {
    const int needed = MultiByteToWideChar(cp, 0, s.c_str(), -1, nullptr, 0);
    if (needed <= 0) return std::wstring();
    std::wstring w(static_cast<size_t>(needed - 1), L'\0');
    MultiByteToWideChar(cp, 0, s.c_str(), -1, w.data(), needed);
    return w;
  };

  std::wstring w = convert(CP_UTF8);
  if (!w.empty()) return w;
  return convert(CP_ACP);
}
#endif

static int32_t SecToFrames(float sec) {
  // 10ms 为一帧
  return static_cast<int32_t>(std::lround(sec * 100.0f));
}

struct RvcEngine::Impl {
  // ORT
  // 说明：这里把日志级别设为 FATAL，避免 ORT 在 stderr 打印大量错误日志（我们会把异常信息回传给宿主）。
  Ort::Env env{ORT_LOGGING_LEVEL_FATAL, "rvc_sdk_ort"};
  Ort::SessionOptions so;
  std::unique_ptr<Ort::Session> sess_encoder;
  std::unique_ptr<Ort::Session> sess_synth;
  std::unique_ptr<Ort::Session> sess_rmvpe;

  std::vector<std::string> enc_in_names;
  std::vector<std::string> enc_out_names;
  std::vector<std::string> syn_in_names;
  std::vector<std::string> syn_out_names;
  std::vector<std::string> rmvpe_in_names;
  std::vector<std::string> rmvpe_out_names;

  // synthesizer onnx 的推理模式：
  // - full: 输出整窗波形（需要在 C++ 侧裁剪 skip_head/return）且 rnd 的时间维度为 T
  // - stream: 输出已裁剪波形（等于 return_length）且 rnd 的时间维度为 T_flow = T - max(skip_head-24,0)
  bool synth_stream = false;

  // FAISS
  std::unique_ptr<faiss::Index> index;
  std::vector<float> big_npy;   // [ntotal, vec_dim]
  int64_t ntotal = 0;

  // SOLA
  std::unique_ptr<Sola> sola;

  // RMVPE（可选）
  std::unique_ptr<RmvpeF0> rmvpe;
};

RvcEngine::RvcEngine(const rvc_sdk_ort_config_t& cfg)
    : cfg_(cfg),
      rng_(114514u),
      norm01_(0.0f, 1.0f),
      impl_(std::make_unique<Impl>()) {
  cache_pitch_.assign(1024, 1);
  cache_pitchf_.assign(1024, 0.0f);
}

RvcEngine::~RvcEngine() = default;

int32_t RvcEngine::GetBlockSize() const {
  return plan_.block_size_io;
}

void RvcEngine::GetRuntimeInfo(rvc_sdk_ort_runtime_info_t* out_info) const {
  if (!out_info) return;
  out_info->io_sample_rate = plan_.io_sr;
  out_info->model_sample_rate = plan_.model_sr;
  out_info->block_size = plan_.block_size_io;
  out_info->total_frames = plan_.total_frames;
  out_info->return_frames = plan_.return_frames;
  out_info->skip_head_frames = plan_.skip_head_frames;
  out_info->synth_stream = (impl_ && impl_->synth_stream) ? 1 : 0;
}

void RvcEngine::ResetState() {
  // 说明：该函数用于“静音 -> 开始说话”等场景下的状态清理，避免旧窗口残留影响新语音。
  std::fill(input_io_.begin(), input_io_.end(), 0.0f);
  std::fill(input_16k_.begin(), input_16k_.end(), 0.0f);
  std::fill(cache_pitch_.begin(), cache_pitch_.end(), 1);
  std::fill(cache_pitchf_.begin(), cache_pitchf_.end(), 0.0f);
  if (impl_ && impl_->sola) {
    impl_->sola->Reset();
  }
  // rng 复位：保证复位后输出更可预期（不会因为历史随机序列导致突发噪声）
  rng_.seed(114514u);
}

bool RvcEngine::InitPlan_(Error* err) {
  if (!CheckDivisibleBy100(cfg_.io_sample_rate)) {
    SetError(err, 1, "io_sample_rate must be divisible by 100.");
    return false;
  }
  if (!CheckDivisibleBy100(cfg_.model_sample_rate)) {
    SetError(err, 2, "model_sample_rate must be divisible by 100.");
    return false;
  }
  if (cfg_.vec_dim != 256 && cfg_.vec_dim != 768) {
    SetError(err, 3, "vec_dim must be 256 or 768.");
    return false;
  }
  if (cfg_.block_time_sec <= 0.0f) {
    SetError(err, 4, "block_time_sec must be > 0.");
    return false;
  }

  plan_.io_sr = cfg_.io_sample_rate;
  plan_.model_sr = cfg_.model_sample_rate;
  plan_.zc_io = plan_.io_sr / 100;
  plan_.zc_model = plan_.model_sr / 100;

  plan_.extra_frames = std::max<int32_t>(0, SecToFrames(cfg_.extra_sec));
  plan_.block_frames = std::max<int32_t>(1, SecToFrames(cfg_.block_time_sec));
  plan_.crossfade_frames = std::max<int32_t>(0, SecToFrames(cfg_.crossfade_sec));
  plan_.sola_buffer_frames = std::min<int32_t>(plan_.crossfade_frames, 4);
  plan_.sola_search_frames = 1;

  plan_.total_frames = plan_.extra_frames + plan_.block_frames + plan_.sola_buffer_frames + plan_.sola_search_frames;
  plan_.return_frames = plan_.block_frames + plan_.sola_buffer_frames + plan_.sola_search_frames;
  plan_.skip_head_frames = plan_.extra_frames;

  plan_.block_size_io = plan_.block_frames * plan_.zc_io;
  plan_.total_size_io = plan_.total_frames * plan_.zc_io;
  plan_.total_size_16k = plan_.total_frames * 160;

  plan_.sola_buffer_size_io = plan_.sola_buffer_frames * plan_.zc_io;
  plan_.sola_search_size_io = plan_.sola_search_frames * plan_.zc_io;

  if (plan_.block_size_io <= plan_.sola_buffer_size_io) {
    SetError(err, 5, "block_time_sec is too small for current crossfade_sec (block_size <= sola_buffer_size).");
    return false;
  }

  input_io_.assign(plan_.total_size_io, 0.0f);
  input_16k_.assign(plan_.total_size_16k, 0.0f);

  sola_buffer_.assign(plan_.sola_buffer_size_io, 0.0f);
  BuildFadeWindows(plan_.sola_buffer_size_io, &fade_in_win_, &fade_out_win_);

  impl_->sola = std::make_unique<Sola>(SolaConfig{plan_.sola_buffer_size_io, plan_.sola_search_size_io});

  return true;
}

static void CollectIO(Ort::Session& sess, std::vector<std::string>* in, std::vector<std::string>* out) {
  Ort::AllocatorWithDefaultOptions alloc;
  const size_t ni = sess.GetInputCount();
  const size_t no = sess.GetOutputCount();
  in->clear();
  out->clear();
  in->reserve(ni);
  out->reserve(no);
  for (size_t i = 0; i < ni; ++i) {
    auto s = sess.GetInputNameAllocated(i, alloc);
    in->push_back(s.get());
  }
  for (size_t i = 0; i < no; ++i) {
    auto s = sess.GetOutputNameAllocated(i, alloc);
    out->push_back(s.get());
  }
}

static bool TryGetIntMetadata(Ort::Session& sess, const char* key, int32_t* out_v) {
  if (!key || !out_v) return false;
  try {
    Ort::AllocatorWithDefaultOptions alloc;
    Ort::ModelMetadata meta = sess.GetModelMetadata();
    auto v = meta.LookupCustomMetadataMapAllocated(key, alloc);
    if (!v) return false;
    const char* s = v.get();
    if (!s || s[0] == '\0') return false;
    char* end = nullptr;
    const long x = std::strtol(s, &end, 10);
    if (end == s) return false;
    *out_v = static_cast<int32_t>(x);
    return true;
  } catch (...) {
    return false;
  }
}

bool RvcEngine::InitSessions_(Error* err) {
  // SessionOptions
  impl_->so = Ort::SessionOptions();
  impl_->so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
  if (cfg_.intra_op_num_threads > 0) {
    impl_->so.SetIntraOpNumThreads(cfg_.intra_op_num_threads);
  }

  if (cfg_.ep == RVC_SDK_ORT_EP_CUDA) {
#ifdef _WIN32
    // 说明：不要在 link-time 直接引用 CUDA/DML 的 append 函数。
    // - CUDA: onnxruntime-win-x64-gpu 的 onnxruntime.dll 导出 OrtSessionOptionsAppendExecutionProvider_CUDA
    // - DML : onnxruntime-directml 的 onnxruntime.dll 导出 OrtSessionOptionsAppendExecutionProvider_DML
    // 两种 onnxruntime.dll 的导出集合不同，直链会导致“换 DLL 就启动失败”。
    using AppendFn = OrtStatus*(ORT_API_CALL*)(OrtSessionOptions*, int);
    auto* fn = reinterpret_cast<AppendFn>(GetOrtExportOrNull_("OrtSessionOptionsAppendExecutionProvider_CUDA"));
    if (!fn) {
      SetError(err,
               10,
               "CUDA EP is not supported by current onnxruntime.dll (missing export OrtSessionOptionsAppendExecutionProvider_CUDA). "
               "Please use onnxruntime-win-x64-gpu build, and ensure onnxruntime_providers_cuda.dll + "
               "onnxruntime_providers_shared.dll are next to the executable (or in PATH).");
      return false;
    }
    const int device_id = 0;
    OrtStatus* st = fn(impl_->so, device_id);
    if (st != nullptr) {
      // 把 ORT 的错误信息带出来，方便定位缺少哪些 DLL（常见：onnxruntime_providers_shared.dll / CUDA / cuDNN）。
      const char* msg = Ort::GetApi().GetErrorMessage(st);
      std::string detail = msg ? msg : "";
      Ort::GetApi().ReleaseStatus(st);
      std::string full = "Failed to enable CUDA EP in onnxruntime (OrtSessionOptionsAppendExecutionProvider_CUDA).";
      if (!detail.empty()) full += " detail=" + detail;
      SetError(err, 10, full);
      return false;
    }
#else
    SetError(err, 10, "CUDA EP is only supported on Windows build in this repo.");
    return false;
#endif
  } else if (cfg_.ep == RVC_SDK_ORT_EP_DML) {
#ifdef _WIN32
    using AppendFn = OrtStatus*(ORT_API_CALL*)(OrtSessionOptions*, int);
    auto* fn = reinterpret_cast<AppendFn>(GetOrtExportOrNull_("OrtSessionOptionsAppendExecutionProvider_DML"));
    if (!fn) {
      SetError(err,
               11,
               "DirectML EP is not supported by current onnxruntime.dll (missing export OrtSessionOptionsAppendExecutionProvider_DML). "
               "Please use Microsoft.ML.OnnxRuntime.DirectML runtime (onnxruntime.dll) next to the executable.");
      return false;
    }
    const int device_id = 0;
    OrtStatus* st = fn(impl_->so, device_id);
    if (st != nullptr) {
      const char* msg = Ort::GetApi().GetErrorMessage(st);
      std::string detail = msg ? msg : "";
      Ort::GetApi().ReleaseStatus(st);
      std::string full = "Failed to enable DirectML EP in onnxruntime (OrtSessionOptionsAppendExecutionProvider_DML).";
      if (!detail.empty()) full += " detail=" + detail;
      SetError(err, 11, full);
      return false;
    }
#else
    SetError(err, 11, "DirectML EP is only supported on Windows.");
    return false;
#endif
  }

  return true;
}

bool RvcEngine::Load(const std::string& content_encoder_onnx,
                     const std::string& synthesizer_onnx,
                     const std::string& faiss_index,
                     Error* err) {
  if (!InitPlan_(err)) return false;
  if (!InitSessions_(err)) return false;

  try {
#ifdef _WIN32
    // 说明：Windows 下 ORTCHAR_T 默认为 wchar_t，Ort::Session 构造函数接收宽字符路径。
    const std::wstring enc_w = Utf8OrAcpToWide_(content_encoder_onnx);
    const std::wstring syn_w = Utf8OrAcpToWide_(synthesizer_onnx);
    impl_->sess_encoder = std::make_unique<Ort::Session>(impl_->env, enc_w.c_str(), impl_->so);
    impl_->sess_synth = std::make_unique<Ort::Session>(impl_->env, syn_w.c_str(), impl_->so);
#else
    impl_->sess_encoder = std::make_unique<Ort::Session>(impl_->env, content_encoder_onnx.c_str(), impl_->so);
    impl_->sess_synth = std::make_unique<Ort::Session>(impl_->env, synthesizer_onnx.c_str(), impl_->so);
#endif
  } catch (const Ort::Exception& e) {
    SetError(err, 20, std::string("Failed to create ORT sessions: ") + e.what());
    return false;
  }

  CollectIO(*impl_->sess_encoder, &impl_->enc_in_names, &impl_->enc_out_names);
  CollectIO(*impl_->sess_synth, &impl_->syn_in_names, &impl_->syn_out_names);

  if (impl_->enc_in_names.size() < 1 || impl_->enc_out_names.size() < 1) {
    SetError(err, 21, "content encoder onnx must have at least 1 input and 1 output.");
    return false;
  }
  if (impl_->syn_in_names.size() < 6 || impl_->syn_out_names.size() < 1) {
    SetError(err, 22, "synthesizer onnx must have 6 inputs and at least 1 output.");
    return false;
  }

  // 优先读取 stream onnx 导出时写入的元数据：
  // - 避免“配置不匹配时 probe 直接在图内报 Reshape 错误”
  // - 同时能给出更明确的提示（stream onnx 基本是固定配置导出）
  int32_t meta_skip = -1;
  int32_t meta_ret = -1;
  const bool has_meta_skip = TryGetIntMetadata(*impl_->sess_synth, "rvc_stream_skip_head_frames", &meta_skip);
  const bool has_meta_ret = TryGetIntMetadata(*impl_->sess_synth, "rvc_stream_return_length_frames", &meta_ret);
  if (has_meta_skip && has_meta_ret) {
    if (meta_skip != plan_.skip_head_frames || meta_ret != plan_.return_frames) {
      std::string msg = "stream synthesizer onnx metadata mismatch. ";
      msg += "export_skip_head_frames=" + std::to_string(meta_skip);
      msg += " export_return_length_frames=" + std::to_string(meta_ret);
      msg += " ; runtime_skip_head_frames=" + std::to_string(plan_.skip_head_frames);
      msg += " runtime_return_frames=" + std::to_string(plan_.return_frames);
      msg += ". Please run with matching --block-sec/--extra-sec/--crossfade-sec, or re-export stream onnx.";
      SetError(err, 24, msg);
      return false;
    }
    impl_->synth_stream = true;
  } else {
    // 无元数据：回退到 probe（兼容旧模型/普通模型）
    if (!DetectSynthMode_(err)) {
      return false;
    }
  }

  // FAISS index
  try {
    faiss::Index* idx = faiss::read_index(faiss_index.c_str());
    if (!idx) {
      SetError(err, 30, "Failed to read faiss index.");
      return false;
    }
    impl_->index.reset(idx);
    impl_->ntotal = impl_->index->ntotal;
    if (impl_->ntotal <= 0) {
      SetError(err, 31, "faiss index is empty.");
      return false;
    }

    // reconstruct_n: [ntotal, vec_dim]
    const int32_t dim = impl_->index->d;
    if (dim != cfg_.vec_dim) {
      SetError(err, 32, "faiss index dim != vec_dim (check your content encoder / model / index match).");
      return false;
    }
    impl_->big_npy.resize(static_cast<size_t>(impl_->ntotal) * static_cast<size_t>(dim));
    impl_->index->reconstruct_n(0, impl_->ntotal, impl_->big_npy.data());
  } catch (const std::exception& e) {
    SetError(err, 33, std::string("Failed to load faiss index: ") + e.what());
    return false;
  }

  return true;
}

bool RvcEngine::LoadRmvpe(const std::string& rmvpe_onnx, Error* err) {
  if (!impl_) {
    SetError(err, 90, "engine impl is null.");
    return false;
  }
  if (rmvpe_onnx.empty()) {
    SetError(err, 91, "rmvpe_onnx path is empty.");
    return false;
  }
  if (!impl_->sess_encoder || !impl_->sess_synth) {
    // 说明：为了减少接口复杂度，这里要求先调用 rvc_sdk_ort_load() 初始化 ORT 环境与基本模型。
    SetError(err, 92, "load rmvpe requires encoder/synth sessions (call rvc_sdk_ort_load first).");
    return false;
  }

  try {
#ifdef _WIN32
    const std::wstring rmvpe_w = Utf8OrAcpToWide_(rmvpe_onnx);
    impl_->sess_rmvpe = std::make_unique<Ort::Session>(impl_->env, rmvpe_w.c_str(), impl_->so);
#else
    impl_->sess_rmvpe = std::make_unique<Ort::Session>(impl_->env, rmvpe_onnx.c_str(), impl_->so);
#endif
  } catch (const Ort::Exception& e) {
    SetError(err, 93, std::string("Failed to create RMVPE ORT session: ") + e.what());
    return false;
  }

  CollectIO(*impl_->sess_rmvpe, &impl_->rmvpe_in_names, &impl_->rmvpe_out_names);
  if (impl_->rmvpe_in_names.empty() || impl_->rmvpe_out_names.empty()) {
    SetError(err, 94, "rmvpe.onnx must have at least 1 input and 1 output.");
    return false;
  }

  impl_->rmvpe = std::make_unique<RmvpeF0>();
  if (!impl_->rmvpe->Init(impl_->sess_rmvpe.get(), err)) {
    // err 已填充
    return false;
  }

  return true;
}

static bool RunSynthProbe_(Ort::Session* sess,
                           const std::vector<std::string>& syn_in_names,
                           const std::vector<std::string>& syn_out_names,
                           int32_t vec_dim,
                           int32_t p_len,
                           int32_t rnd_len,
                           size_t* out_n,
                           std::string* out_err) {
  if (!sess || syn_in_names.size() < 6 || syn_out_names.empty()) return false;
  if (p_len <= 0 || rnd_len <= 0) return false;

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const int64_t B = 1;
  const int64_t T = static_cast<int64_t>(p_len);
  const int64_t C = static_cast<int64_t>(vec_dim);
  const int64_t Tr = static_cast<int64_t>(rnd_len);

  std::vector<float> phone(static_cast<size_t>(T) * static_cast<size_t>(C), 0.0f);
  std::vector<int64_t> phone_lengths{static_cast<int64_t>(T)};
  std::vector<int64_t> pitch(static_cast<size_t>(T), 1);
  std::vector<float> pitchf(static_cast<size_t>(T), 0.0f);
  std::vector<int64_t> sid{0};
  std::vector<float> rnd(static_cast<size_t>(192) * static_cast<size_t>(Tr), 0.0f);

  std::vector<int64_t> phone_shape{B, T, C};
  std::vector<int64_t> len_shape{B};
  std::vector<int64_t> pitch_shape{B, T};
  std::vector<int64_t> pitchf_shape{B, T};
  std::vector<int64_t> sid_shape{B};
  std::vector<int64_t> rnd_shape{B, 192, Tr};

  Ort::Value phone_t = Ort::Value::CreateTensor<float>(mem,
                                                       phone.data(),
                                                       phone.size(),
                                                       phone_shape.data(),
                                                       phone_shape.size());
  Ort::Value len_t = Ort::Value::CreateTensor<int64_t>(mem,
                                                       phone_lengths.data(),
                                                       phone_lengths.size(),
                                                       len_shape.data(),
                                                       len_shape.size());
  Ort::Value pitch_t = Ort::Value::CreateTensor<int64_t>(mem,
                                                         pitch.data(),
                                                         pitch.size(),
                                                         pitch_shape.data(),
                                                         pitch_shape.size());
  Ort::Value pitchf_t = Ort::Value::CreateTensor<float>(mem,
                                                        pitchf.data(),
                                                        pitchf.size(),
                                                        pitchf_shape.data(),
                                                        pitchf_shape.size());
  Ort::Value sid_t = Ort::Value::CreateTensor<int64_t>(mem, sid.data(), sid.size(), sid_shape.data(), sid_shape.size());
  Ort::Value rnd_t = Ort::Value::CreateTensor<float>(mem, rnd.data(), rnd.size(), rnd_shape.data(), rnd_shape.size());

  const char* in_names[6] = {
      syn_in_names[0].c_str(),
      syn_in_names[1].c_str(),
      syn_in_names[2].c_str(),
      syn_in_names[3].c_str(),
      syn_in_names[4].c_str(),
      syn_in_names[5].c_str(),
  };
  const char* out_names[] = {syn_out_names[0].c_str()};

  Ort::Value inputs[] = {std::move(phone_t),
                         std::move(len_t),
                         std::move(pitch_t),
                         std::move(pitchf_t),
                         std::move(sid_t),
                         std::move(rnd_t)};

  try {
    std::vector<Ort::Value> out = sess->Run(Ort::RunOptions{nullptr}, in_names, inputs, 6, out_names, 1);
    if (out.empty()) return false;
    auto info = out[0].GetTensorTypeAndShapeInfo();
    const size_t n = info.GetElementCount();
    if (out_n) *out_n = n;
    return n > 0;
  } catch (const Ort::Exception& e) {
    if (out_err) *out_err = e.what();
    return false;
  }
}

bool RvcEngine::DetectSynthMode_(Error* err) {
  if (!impl_->sess_synth) {
    SetError(err, 23, "synthesizer session is null.");
    return false;
  }

  const int32_t T = plan_.total_frames;
  const int32_t flow_head = std::max<int32_t>(plan_.skip_head_frames - 24, 0);
  const int32_t T_flow = std::max<int32_t>(T - flow_head, 1);

  // 输出长度（以 model_sr 为基准）
  const size_t expect_full = static_cast<size_t>(plan_.total_frames) * static_cast<size_t>(plan_.zc_model);
  const size_t expect_stream = static_cast<size_t>(plan_.return_frames) * static_cast<size_t>(plan_.zc_model);

  // 说明：当 skip_head < 24 时，T_flow == T，单纯通过 rnd_len 无法区分 full/stream。
  // 因此这里以“输出长度”作为主判据（stream 导出只输出 return_length）。
  size_t n0 = 0;
  std::string e0;
  const bool ok0 = RunSynthProbe_(impl_->sess_synth.get(),
                                 impl_->syn_in_names,
                                 impl_->syn_out_names,
                                 cfg_.vec_dim,
                                 T,
                                 T,
                                 &n0,
                                 &e0);
  if (ok0) {
    if (n0 == expect_stream) {
      impl_->synth_stream = true;
      return true;
    }
    if (n0 >= expect_full) {
      impl_->synth_stream = false;
      return true;
    }
    // ok0 但长度不匹配：继续尝试 T_flow（用于 skip_head>=24 或某些导出差异）
  }

  size_t n1 = 0;
  std::string e1;
  const bool ok1 = RunSynthProbe_(impl_->sess_synth.get(),
                                 impl_->syn_in_names,
                                 impl_->syn_out_names,
                                 cfg_.vec_dim,
                                 T,
                                 T_flow,
                                 &n1,
                                 &e1);
  if (ok1) {
    if (n1 == expect_stream) {
      impl_->synth_stream = true;
      return true;
    }
    if (n1 >= expect_full) {
      impl_->synth_stream = false;
      return true;
    }
  }

  if (ok0 || ok1) {
    std::string msg = "synthesizer output length mismatch (check your model_sr/block/extra/crossfade settings, "
                      "or re-export stream onnx to match current config).";
    msg += " n0=" + std::to_string(n0);
    msg += " n1=" + std::to_string(n1);
    msg += " expect_full=" + std::to_string(expect_full);
    msg += " expect_stream=" + std::to_string(expect_stream);
    msg += " T=" + std::to_string(T);
    msg += " T_flow=" + std::to_string(T_flow);
    SetError(err, 24, msg);
    return false;
  }

  SetError(err, 25, std::string("failed to probe synthesizer mode. probe0_err=") + e0 + " ; probe1_err=" + e1);
  return false;
}

void RvcEngine::UpdateInputBuffers_(const float* in_mono, int32_t in_frames) {
  // input_io_：左移 block_size_io，末尾写入 in_mono
  const int32_t B = plan_.block_size_io;
  const int32_t N = plan_.total_size_io;
  if (B <= 0 || N <= 0) return;
  // in_frames 必须等于 B
  std::memmove(input_io_.data(), input_io_.data() + B, sizeof(float) * (N - B));
  std::memcpy(input_io_.data() + (N - B), in_mono, sizeof(float) * B);

  // 16k 缓冲：把本 block 重采样到 block_frames*160，然后左移追加
  const int32_t B16 = plan_.block_frames * 160;
  const int32_t N16 = plan_.total_size_16k;
  std::vector<float> tmp16(B16);
  ResampleLinear(in_mono, B, tmp16.data(), B16);
  std::memmove(input_16k_.data(), input_16k_.data() + B16, sizeof(float) * (N16 - B16));
  std::memcpy(input_16k_.data() + (N16 - B16), tmp16.data(), sizeof(float) * B16);
}

static void F0ToCoarse(const std::vector<float>& f0_hz,
                       int32_t p_len,
                       float f0_min,
                       float f0_max,
                       std::vector<int64_t>* pitch,
                       std::vector<float>* pitchf) {
  pitch->resize(p_len);
  pitchf->resize(p_len);

  const float f0_mel_min = 1127.0f * std::log(1.0f + f0_min / 700.0f);
  const float f0_mel_max = 1127.0f * std::log(1.0f + f0_max / 700.0f);

  for (int32_t i = 0; i < p_len; ++i) {
    const float f0 = (i < static_cast<int32_t>(f0_hz.size())) ? f0_hz[i] : 0.0f;
    (*pitchf)[i] = f0;
    if (f0 <= 0.0f) {
      (*pitch)[i] = 1;
      continue;
    }
    float f0_mel = 1127.0f * std::log(1.0f + f0 / 700.0f);
    f0_mel = (f0_mel - f0_mel_min) * 254.0f / (f0_mel_max - f0_mel_min) + 1.0f;
    if (f0_mel <= 1.0f) f0_mel = 1.0f;
    if (f0_mel > 255.0f) f0_mel = 255.0f;
    (*pitch)[i] = static_cast<int64_t>(std::llround(f0_mel));
  }
}

bool RvcEngine::ComputeF0WithCache_(int32_t p_len,
                                   std::vector<int64_t>* pitch,
                                   std::vector<float>* pitchf,
                                   Error* err) {
  // 计算最近一段的 f0，然后按 tools/rvc_for_realtime.py 的 cache 逻辑更新。
  //
  // 说明：
  // - 本 SDK 支持两种 F0：YIN（简单可用）与 RMVPE（通常更稳）。
  // - segment_len 参考 python：block_frames*160 + 800（与 tools/rvc_for_realtime.py 一致的“额外上下文”）
  // - 输出长度习惯：segment_len/160 + 1（10ms 一帧）

  const int32_t audio_len = static_cast<int32_t>(input_16k_.size());
  // 说明：
  // - 当 block 很小（例如 0.05~0.10s）时，`block_frames*160 + 800` 可能只有 0.1~0.2s，
  //   YIN 会更容易抖动，进而导致“电音/喘声/胡言乱语”等伪影。
  // - 这里给一个最小上下文（0.30s @ 16k = 4800 samples），在总缓冲足够时自动加长分析片段，
  //   在不显著增加延迟的前提下提升短 block 的稳定性。
  const int32_t base_seg_len = plan_.block_frames * 160 + 800;
  const int32_t min_seg_len = 4800;
  int32_t seg_len = std::min<int32_t>(audio_len, std::max<int32_t>(base_seg_len, min_seg_len));

  // 对齐 python：rmvpe 常把音频长度对齐到 5120 的倍数（32 帧）以减少 padding 与边界不稳定。
  if (cfg_.f0_method == RVC_SDK_ORT_F0_RMVPE) {
    const int32_t want = std::max<int32_t>(base_seg_len, min_seg_len);
    // 5120 samples = 32 frames * 160 hop；减 160 是为了让 n_frames = len/160 + 1 恰好是 32 的倍数。
    const int32_t aligned = 5120 * ((want - 1) / 5120 + 1) - 160;
    if (aligned > 0) {
      seg_len = std::min<int32_t>(audio_len, aligned);
    }
  }

  const int32_t seg_start = std::max<int32_t>(0, audio_len - seg_len);

  std::vector<float> seg_f0;
  if (cfg_.f0_method == RVC_SDK_ORT_F0_RMVPE) {
    if (!impl_ || !impl_->rmvpe) {
      SetError(err, 56, "F0 method is RMVPE but rmvpe.onnx is not loaded (call rvc_sdk_ort_load_rmvpe).");
      return false;
    }
    const float th = (cfg_.rmvpe_threshold > 0.0f) ? cfg_.rmvpe_threshold : 0.03f;
    if (!impl_->rmvpe->ComputeF0Hz(input_16k_.data() + seg_start, seg_len, th, &seg_f0, err)) {
      return false;
    }
    // RMVPE 输出是全局范围，额外按 f0_min/f0_max 做一次裁剪，避免极端值影响后续模型。
    const float f0_min = cfg_.f0_min_hz > 0.0f ? cfg_.f0_min_hz : 50.0f;
    const float f0_max = cfg_.f0_max_hz > 0.0f ? cfg_.f0_max_hz : 1100.0f;
    for (float& v : seg_f0) {
      if (v < f0_min || v > f0_max) v = 0.0f;
    }
  } else {
    ComputeF0Yin16k(input_16k_.data() + seg_start,
                    seg_len,
                    16000,
                    160,
                    cfg_.f0_min_hz > 0.0f ? cfg_.f0_min_hz : 50.0f,
                    cfg_.f0_max_hz > 0.0f ? cfg_.f0_max_hz : 1100.0f,
                    &seg_f0);
  }

  // 变调
  const float ratio = SemitoneToRatio(cfg_.f0_up_key);
  for (float& v : seg_f0) {
    v = v > 0.0f ? (v * ratio) : 0.0f;
  }

  // 转 coarse
  std::vector<int64_t> seg_pitch;
  std::vector<float> seg_pitchf;
  const int32_t seg_p_len = static_cast<int32_t>(seg_f0.size());
  F0ToCoarse(seg_f0, seg_p_len,
             cfg_.f0_min_hz > 0.0f ? cfg_.f0_min_hz : 50.0f,
             cfg_.f0_max_hz > 0.0f ? cfg_.f0_max_hz : 1100.0f,
             &seg_pitch, &seg_pitchf);

  const int32_t shift = plan_.block_frames;
  if (shift > 0 && shift < static_cast<int32_t>(cache_pitch_.size())) {
    std::memmove(cache_pitch_.data(), cache_pitch_.data() + shift, sizeof(int64_t) * (cache_pitch_.size() - shift));
    std::memmove(cache_pitchf_.data(), cache_pitchf_.data() + shift, sizeof(float) * (cache_pitchf_.size() - shift));
  }

  // 对齐 python：cache_pitch[4 - L :] = pitch[3:-1]
  // 目的：丢掉边界不稳定部分，并让最新帧落在 cache 尾部。
  const int32_t L = seg_p_len;
  if (L >= 5) {
    const int32_t copy_len = L - 4;
    const int32_t dst_start = static_cast<int32_t>(cache_pitch_.size()) - copy_len;
    for (int32_t i = 0; i < copy_len; ++i) {
      cache_pitch_[dst_start + i] = seg_pitch[3 + i];
      cache_pitchf_[dst_start + i] = seg_pitchf[3 + i];
    }
  } else {
    // 太短时，直接不更新（保持上一次 cache）
  }

  // 取 cache 的最后 p_len 帧作为输入
  if (p_len > static_cast<int32_t>(cache_pitch_.size())) {
    // p_len 异常大，直接扩展（避免崩溃）
    cache_pitch_.resize(p_len, 1);
    cache_pitchf_.resize(p_len, 0.0f);
  }
  const int32_t start = static_cast<int32_t>(cache_pitch_.size()) - p_len;
  pitch->assign(cache_pitch_.begin() + start, cache_pitch_.end());
  pitchf->assign(cache_pitchf_.begin() + start, cache_pitchf_.end());
  return true;
}

static bool NormalizeFeatsShape(const Ort::Value& out,
                                int32_t vec_dim,
                                std::vector<float>* feats20,  // [T20, C]
                                int32_t* t20,
                                Error* err) {
  // 允许输出形状：
  // - [1, C, T] 或 [1, T, C] 或 [C, T] 或 [T, C]
  // 最终转为 row-major: [T, C]
  if (!feats20 || !t20) return false;
  feats20->clear();
  *t20 = 0;

  auto type_info = out.GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = type_info.GetShape();
  if (shape.empty()) {
    SetError(err, 40, "content encoder output shape is empty.");
    return false;
  }

  // 扁平取数据
  const float* data = out.GetTensorData<float>();

  // 去掉 batch=1
  if (shape.size() == 3 && shape[0] == 1) {
    shape.erase(shape.begin());
  }

  if (shape.size() != 2) {
    SetError(err, 41, "content encoder output must be 2D or 3D with batch=1.");
    return false;
  }

  const int64_t a = shape[0];
  const int64_t b = shape[1];
  if (a == vec_dim) {
    // [C, T] -> [T, C]
    *t20 = static_cast<int32_t>(b);
    feats20->resize(static_cast<size_t>(*t20) * static_cast<size_t>(vec_dim));
    for (int32_t t = 0; t < *t20; ++t) {
      for (int32_t c = 0; c < vec_dim; ++c) {
        (*feats20)[static_cast<size_t>(t) * vec_dim + c] = data[static_cast<size_t>(c) * (*t20) + t];
      }
    }
    return true;
  }
  if (b == vec_dim) {
    // [T, C] 直接拷贝
    *t20 = static_cast<int32_t>(a);
    feats20->assign(data, data + static_cast<size_t>(a) * static_cast<size_t>(b));
    return true;
  }

  SetError(err, 42, "content encoder output dims do not match vec_dim.");
  return false;
}

bool RvcEngine::ExtractAndRetrieve_(std::vector<float>* feats10, int32_t* p_len, Error* err) {
  if (!feats10 || !p_len) return false;
  if (!impl_->sess_encoder || !impl_->sess_synth || !impl_->index) {
    SetError(err, 50, "engine not loaded.");
    return false;
  }

  // p_len（10ms 帧数）由缓冲长度决定
  *p_len = plan_.total_frames;

  // ---- 1) encoder ONNX：输入 16k wav ----
  // 输入形状通常为 [1, 1, T]
  Ort::AllocatorWithDefaultOptions alloc;
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  const int64_t T = static_cast<int64_t>(input_16k_.size());
  std::vector<int64_t> in_shape{1, 1, T};
  Ort::Value in_tensor = Ort::Value::CreateTensor<float>(
      mem,
      const_cast<float*>(input_16k_.data()),
      input_16k_.size(),
      in_shape.data(),
      in_shape.size());

  const char* in_names[] = {impl_->enc_in_names[0].c_str()};
  const char* out_names[] = {impl_->enc_out_names[0].c_str()};

  std::vector<Ort::Value> enc_out;
  try {
    enc_out = impl_->sess_encoder->Run(Ort::RunOptions{nullptr}, in_names, &in_tensor, 1, out_names, 1);
  } catch (const Ort::Exception& e) {
    SetError(err, 51, std::string("encoder run failed: ") + e.what());
    return false;
  }
  if (enc_out.empty()) {
    SetError(err, 52, "encoder produced no outputs.");
    return false;
  }

  // 归一化输出形状到 feats20: [T20, C]
  std::vector<float> feats20;
  int32_t t20 = 0;
  if (!NormalizeFeatsShape(enc_out[0], cfg_.vec_dim, &feats20, &t20, err)) {
    return false;
  }
  if (t20 <= 0) {
    SetError(err, 53, "encoder output T is invalid.");
    return false;
  }

  // mimic torch: feats = cat(feats, last)
  feats20.resize(static_cast<size_t>(t20 + 1) * static_cast<size_t>(cfg_.vec_dim));
  std::memcpy(feats20.data() + static_cast<size_t>(t20) * cfg_.vec_dim,
              feats20.data() + static_cast<size_t>(t20 - 1) * cfg_.vec_dim,
              sizeof(float) * cfg_.vec_dim);
  t20 += 1;

  // ---- 2) FAISS 检索融合（在 20ms 帧上做） ----
  const int32_t skip20 = plan_.skip_head_frames / 2;
  if (cfg_.index_rate > 0.0f && skip20 < t20) {
    const int32_t n = t20 - skip20;
    const int32_t k = 8;
    std::vector<float> distances(static_cast<size_t>(n) * k);
    std::vector<faiss::idx_t> labels(static_cast<size_t>(n) * k);

    // query 指向 feats20[skip20:]
    const float* query = feats20.data() + static_cast<size_t>(skip20) * cfg_.vec_dim;
    impl_->index->search(n, query, k, distances.data(), labels.data());

    // 加权融合（与 python 对齐：weight=(1/score)^2，再归一化）
    const float eps = 1e-6f;
    std::vector<float> retrieved(cfg_.vec_dim);
    for (int32_t i = 0; i < n; ++i) {
      bool ok = true;
      for (int32_t j = 0; j < k; ++j) {
        if (labels[static_cast<size_t>(i) * k + j] < 0) {
          ok = false;
          break;
        }
      }
      if (!ok) continue;

      float wsum = 0.0f;
      float w[k];
      for (int32_t j = 0; j < k; ++j) {
        const float d = std::max(distances[static_cast<size_t>(i) * k + j], eps);
        w[j] = Pow2(1.0f / d);
        wsum += w[j];
      }
      if (wsum <= 0.0f) continue;
      for (int32_t j = 0; j < k; ++j) w[j] /= wsum;

      std::fill(retrieved.begin(), retrieved.end(), 0.0f);
      for (int32_t j = 0; j < k; ++j) {
        const auto id = labels[static_cast<size_t>(i) * k + j];
        const float* v = impl_->big_npy.data() + static_cast<size_t>(id) * cfg_.vec_dim;
        const float ww = w[j];
        for (int32_t c = 0; c < cfg_.vec_dim; ++c) {
          retrieved[c] += ww * v[c];
        }
      }

      float* dst = feats20.data() + static_cast<size_t>(skip20 + i) * cfg_.vec_dim;
      for (int32_t c = 0; c < cfg_.vec_dim; ++c) {
        dst[c] = cfg_.index_rate * retrieved[c] + (1.0f - cfg_.index_rate) * dst[c];
      }
    }
  }

  // ---- 3) 上采样到 10ms 帧（2x repeat），并裁剪到 p_len ----
  // 上采样：T10 = T20 * 2
  const int32_t t10 = t20 * 2;
  std::vector<float> tmp10(static_cast<size_t>(t10) * cfg_.vec_dim);
  for (int32_t i = 0; i < t20; ++i) {
    const float* src = feats20.data() + static_cast<size_t>(i) * cfg_.vec_dim;
    float* dst0 = tmp10.data() + static_cast<size_t>(2 * i) * cfg_.vec_dim;
    float* dst1 = tmp10.data() + static_cast<size_t>(2 * i + 1) * cfg_.vec_dim;
    std::memcpy(dst0, src, sizeof(float) * cfg_.vec_dim);
    std::memcpy(dst1, src, sizeof(float) * cfg_.vec_dim);
  }

  // 补齐到 p_len
  if (t10 < *p_len) {
    // 用最后一帧填充
    const float* last = tmp10.data() + static_cast<size_t>(std::max<int32_t>(0, t10 - 1)) * cfg_.vec_dim;
    tmp10.resize(static_cast<size_t>(*p_len) * cfg_.vec_dim);
    for (int32_t i = t10; i < *p_len; ++i) {
      std::memcpy(tmp10.data() + static_cast<size_t>(i) * cfg_.vec_dim, last, sizeof(float) * cfg_.vec_dim);
    }
  }
  // 裁剪到 p_len
  feats10->assign(tmp10.begin(), tmp10.begin() + static_cast<size_t>(*p_len) * cfg_.vec_dim);
  return true;
}

bool RvcEngine::InferWindow_(std::vector<float>* infer_wav_io, Error* err) {
  if (!infer_wav_io) return false;
  infer_wav_io->clear();

  int32_t p_len = 0;
  std::vector<float> feats10;
  if (!ExtractAndRetrieve_(&feats10, &p_len, err)) {
    return false;
  }

  std::vector<int64_t> pitch;
  std::vector<float> pitchf;
  if (!ComputeF0WithCache_(p_len, &pitch, &pitchf, err)) {
    return false;
  }

  // ---- synthesizer ONNX ----
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  const int64_t B = 1;
  const int64_t T = p_len;
  const int64_t C = cfg_.vec_dim;

  std::vector<int64_t> phone_shape{B, T, C};
  std::vector<int64_t> len_shape{B};
  std::vector<int64_t> pitch_shape{B, T};
  std::vector<int64_t> pitchf_shape{B, T};
  std::vector<int64_t> sid_shape{B};

  // 说明：stream synthesizer（infer 路径导出）会在 enc_p(flow_head) 后裁剪时间轴，
  // 因此 rnd 的时间维度必须是 T_flow = T - max(skip_head-24,0)。
  const int32_t flow_head = std::max<int32_t>(plan_.skip_head_frames - 24, 0);
  const int32_t rnd_len = impl_->synth_stream ? std::max<int32_t>(p_len - flow_head, 1) : p_len;
  std::vector<int64_t> rnd_shape{B, 192, static_cast<int64_t>(rnd_len)};

  std::vector<int64_t> phone_lengths(1, p_len);
  std::vector<int64_t> sid(1, cfg_.sid);

  // rnd
  std::vector<float> rnd(static_cast<size_t>(192) * static_cast<size_t>(rnd_len));
  // 与 python 推理对齐：torch.randn_like(...) * noise_scale
  const float noise = (cfg_.noise_scale >= 0.0f) ? cfg_.noise_scale : 0.0f;
  for (float& v : rnd) v = norm01_(rng_) * noise;

  Ort::Value phone_t = Ort::Value::CreateTensor<float>(mem,
                                                       feats10.data(),
                                                       feats10.size(),
                                                       phone_shape.data(),
                                                       phone_shape.size());
  Ort::Value len_t = Ort::Value::CreateTensor<int64_t>(mem,
                                                       phone_lengths.data(),
                                                       phone_lengths.size(),
                                                       len_shape.data(),
                                                       len_shape.size());
  Ort::Value pitch_t = Ort::Value::CreateTensor<int64_t>(mem,
                                                         pitch.data(),
                                                         pitch.size(),
                                                         pitch_shape.data(),
                                                         pitch_shape.size());
  Ort::Value pitchf_t = Ort::Value::CreateTensor<float>(mem,
                                                        pitchf.data(),
                                                        pitchf.size(),
                                                        pitchf_shape.data(),
                                                        pitchf_shape.size());
  Ort::Value sid_t = Ort::Value::CreateTensor<int64_t>(mem,
                                                       sid.data(),
                                                       sid.size(),
                                                       sid_shape.data(),
                                                       sid_shape.size());
  Ort::Value rnd_t = Ort::Value::CreateTensor<float>(mem,
                                                     rnd.data(),
                                                     rnd.size(),
                                                     rnd_shape.data(),
                                                     rnd_shape.size());

  // 输入顺序按本仓库 export 脚本：phone, phone_lengths, pitch, pitchf, ds, rnd
  const char* in_names[6] = {
      impl_->syn_in_names[0].c_str(),
      impl_->syn_in_names[1].c_str(),
      impl_->syn_in_names[2].c_str(),
      impl_->syn_in_names[3].c_str(),
      impl_->syn_in_names[4].c_str(),
      impl_->syn_in_names[5].c_str(),
  };
  const char* out_names[] = {impl_->syn_out_names[0].c_str()};

  Ort::Value inputs[] = {std::move(phone_t),
                         std::move(len_t),
                         std::move(pitch_t),
                         std::move(pitchf_t),
                         std::move(sid_t),
                         std::move(rnd_t)};

  std::vector<Ort::Value> out;
  try {
    out = impl_->sess_synth->Run(Ort::RunOptions{nullptr}, in_names, inputs, 6, out_names, 1);
  } catch (const Ort::Exception& e) {
    SetError(err, 60, std::string("synthesizer run failed: ") + e.what());
    return false;
  }
  if (out.empty()) {
    SetError(err, 61, "synthesizer produced no outputs.");
    return false;
  }

  // 输出波形（float32），形状通常 [1, 1, N] 或 [1, N]
  const float* audio = out[0].GetTensorData<float>();
  auto info = out[0].GetTensorTypeAndShapeInfo();
  const size_t n = info.GetElementCount();
  if (n == 0) {
    SetError(err, 62, "synthesizer output size is invalid.");
    return false;
  }
  std::vector<float> wav_model(audio, audio + static_cast<int64_t>(n));

  const int32_t need = plan_.return_frames * plan_.zc_model;
  std::vector<float> infer_model;
  if (impl_->synth_stream) {
    // stream onnx 已经在图内做过裁剪：输出应严格等于 return_frames 对应的长度。
    if (static_cast<int32_t>(wav_model.size()) != need) {
      SetError(err, 63, "stream synthesizer output length mismatch (check your stream export settings).");
      return false;
    }
    infer_model = std::move(wav_model);
  } else {
    // ---- 裁剪出“skip_head -> skip_head + return_frames”的片段 ----
    const int32_t start = plan_.skip_head_frames * plan_.zc_model;
    if (start + need > static_cast<int32_t>(wav_model.size())) {
      SetError(err, 63, "synthesizer output is shorter than expected (check model_sr / window sizes).");
      return false;
    }
    infer_model.assign(wav_model.begin() + start, wav_model.begin() + start + need);
  }

  // ---- resample model_sr -> io_sr（如果不同） ----
  std::vector<float> infer_io;
  const int32_t need_io = plan_.return_frames * plan_.zc_io;
  if (plan_.model_sr == plan_.io_sr) {
    infer_io = std::move(infer_model);
  } else {
    infer_io.resize(need_io);
    ResampleLinear(infer_model.data(), static_cast<int32_t>(infer_model.size()), infer_io.data(), need_io);
  }

  *infer_wav_io = std::move(infer_io);
  return true;
}

bool RvcEngine::ProcessBlock(const float* in_mono,
                            int32_t in_frames,
                            float* out_mono,
                            int32_t out_frames,
                            Error* err) {
  if (!in_mono || !out_mono) {
    SetError(err, 70, "null input/output.");
    return false;
  }
  if (in_frames != plan_.block_size_io || out_frames != plan_.block_size_io) {
    SetError(err, 71, "block size mismatch (use rvc_sdk_ort_get_block_size).");
    return false;
  }

  UpdateInputBuffers_(in_mono, in_frames);

  std::vector<float> infer_wav;
  if (!InferWindow_(&infer_wav, err)) {
    return false;
  }

  // 音量包络混合（可选）：把输出的 RMS 包络部分拉回输入，减少静音段“喘声/怪声”等伪影。
  // 说明：这里做的是“每 10ms 一段”的简化版本（piecewise constant），避免引入复杂插值与额外依赖。
  if (cfg_.rms_mix_rate >= 0.0f && cfg_.rms_mix_rate < 1.0f) {
    const float mix = std::max<float>(0.0f, std::min<float>(1.0f, cfg_.rms_mix_rate));
    const float exponent = 1.0f - mix;
    const int32_t hop = plan_.zc_io;  // 10ms
    const int32_t n = static_cast<int32_t>(infer_wav.size());
    const int32_t frames = (hop > 0) ? (n / hop) : 0;
    const int32_t in_start = plan_.skip_head_frames * plan_.zc_io;
    const int32_t in_total = static_cast<int32_t>(input_io_.size());
    if (hop > 0 && frames > 0 && frames * hop == n && in_start >= 0 && in_start + n <= in_total) {
      const float* in_ref = input_io_.data() + in_start;
      constexpr float kEps = 1e-3f;
      for (int32_t f = 0; f < frames; ++f) {
        const float* xin = in_ref + f * hop;
        float* xout = infer_wav.data() + f * hop;
        double s2_in = 0.0;
        double s2_out = 0.0;
        for (int32_t i = 0; i < hop; ++i) {
          const float a = xin[i];
          const float b = xout[i];
          s2_in += (double)a * (double)a;
          s2_out += (double)b * (double)b;
        }
        const float rms_in = (hop > 0) ? (float)std::sqrt(s2_in / (double)hop) : 0.0f;
        float rms_out = (hop > 0) ? (float)std::sqrt(s2_out / (double)hop) : 0.0f;
        if (rms_out < kEps) rms_out = kEps;
        const float ratio = rms_in / rms_out;
        // 限制缩放，避免极端情况下爆音/过度衰减
        float scale = std::pow(std::max<float>(ratio, 0.0f), exponent);
        if (scale > 8.0f) scale = 8.0f;
        if (scale < 0.0f) scale = 0.0f;
        for (int32_t i = 0; i < hop; ++i) {
          xout[i] *= scale;
        }
      }
    }
  }

  // infer_wav 长度应为 return_frames*zc_io >= sola_buffer + sola_search + block
  const int32_t block = plan_.block_size_io;
  const int32_t need = plan_.sola_buffer_size_io + plan_.sola_search_size_io + block;
  if (static_cast<int32_t>(infer_wav.size()) < need) {
    SetError(err, 72, "infer_wav is too short for SOLA.");
    return false;
  }

  // SOLA 对齐 + 输出 block
  const int32_t infer_len = static_cast<int32_t>(infer_wav.size());
  const int32_t off = impl_->sola->Process(infer_wav.data(),
                                          infer_len,
                                          block,
                                          fade_in_win_.data(),
                                          fade_out_win_.data(),
                                          out_mono);
  (void)off;  // 调试时可记录

  // 限幅（避免超出 [-1,1]）
  for (int32_t i = 0; i < block; ++i) {
    if (out_mono[i] > 1.0f) out_mono[i] = 1.0f;
    if (out_mono[i] < -1.0f) out_mono[i] = -1.0f;
  }

  return true;
}

}  // namespace rvc_ort
