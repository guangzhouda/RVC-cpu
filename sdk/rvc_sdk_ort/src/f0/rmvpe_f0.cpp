// rmvpe_f0.cpp

#include "f0/rmvpe_f0.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "rvc_engine.h"  // Error

// ORT
#include <onnxruntime_cxx_api.h>

// kissfft（第三方）
extern "C" {
#include "kiss_fftr.h"
}

namespace rvc_ort {
namespace {

static void SetErr_(Error* err, int32_t code, const std::string& msg) {
  if (!err) return;
  err->code = code;
  err->message = msg;
}

static float HzToMelHTK_(float hz) {
  // librosa.filters.mel(htk=True) 的 mel 标度：mel = 2595*log10(1+f/700)
  return 2595.0f * std::log10(1.0f + hz / 700.0f);
}

static float MelToHzHTK_(float mel) {
  return 700.0f * (std::pow(10.0f, mel / 2595.0f) - 1.0f);
}

static void BuildHannWindow_(int32_t win_length, std::vector<float>* out) {
  out->assign(std::max<int32_t>(0, win_length), 0.0f);
  if (win_length <= 0) return;
  if (win_length == 1) {
    (*out)[0] = 1.0f;
    return;
  }
  // 对齐 torch.hann_window（periodic=True）：w[n] = 0.5 - 0.5*cos(2*pi*n/N)
  constexpr float kPi = 3.14159265358979323846f;
  const float N = static_cast<float>(win_length);
  for (int32_t n = 0; n < win_length; ++n) {
    (*out)[n] = 0.5f - 0.5f * std::cos(2.0f * kPi * static_cast<float>(n) / N);
  }
}

static void BuildMelBasisHTK_(int32_t sr,
                              int32_t n_fft,
                              int32_t n_mels,
                              float fmin,
                              float fmax,
                              std::vector<float>* out_basis) {
  const int32_t n_freq = n_fft / 2 + 1;
  out_basis->assign(static_cast<size_t>(n_mels) * static_cast<size_t>(n_freq), 0.0f);

  const float mel_min = HzToMelHTK_(std::max<float>(0.0f, fmin));
  const float mel_max = HzToMelHTK_(std::max<float>(fmin + 1.0f, fmax));

  // mel 点：n_mels + 2
  std::vector<float> mel_pts(static_cast<size_t>(n_mels + 2));
  for (int32_t i = 0; i < n_mels + 2; ++i) {
    const float a = (n_mels + 1) > 0 ? static_cast<float>(i) / static_cast<float>(n_mels + 1) : 0.0f;
    mel_pts[static_cast<size_t>(i)] = mel_min + (mel_max - mel_min) * a;
  }
  std::vector<float> hz_pts(static_cast<size_t>(n_mels + 2));
  for (int32_t i = 0; i < n_mels + 2; ++i) {
    hz_pts[static_cast<size_t>(i)] = MelToHzHTK_(mel_pts[static_cast<size_t>(i)]);
  }

  // FFT bin 频率：0..sr/2，共 n_freq 个点（与 librosa.fft_frequencies 对齐）
  std::vector<float> fft_freq(static_cast<size_t>(n_freq));
  for (int32_t k = 0; k < n_freq; ++k) {
    fft_freq[static_cast<size_t>(k)] = static_cast<float>(sr) * static_cast<float>(k) / static_cast<float>(n_fft);
  }

  // 三角滤波器组
  for (int32_t m = 0; m < n_mels; ++m) {
    const float left = hz_pts[static_cast<size_t>(m)];
    const float center = hz_pts[static_cast<size_t>(m + 1)];
    const float right = hz_pts[static_cast<size_t>(m + 2)];
    const float inv_left = (center > left) ? (1.0f / (center - left)) : 0.0f;
    const float inv_right = (right > center) ? (1.0f / (right - center)) : 0.0f;

    for (int32_t k = 0; k < n_freq; ++k) {
      const float f = fft_freq[static_cast<size_t>(k)];
      float w = 0.0f;
      if (f >= left && f <= center) {
        w = (center > left) ? (f - left) * inv_left : 0.0f;
      } else if (f > center && f <= right) {
        w = (right > center) ? (right - f) * inv_right : 0.0f;
      }
      (*out_basis)[static_cast<size_t>(m) * static_cast<size_t>(n_freq) + static_cast<size_t>(k)] = w;
    }
  }
}

static int32_t ReflectIndex_(int32_t idx, int32_t len) {
  if (len <= 1) return 0;
  if (idx < 0) idx = -idx;  // reflect
  if (idx >= len) idx = (len - 1) - (idx - (len - 1));  // reflect
  if (idx < 0) idx = 0;
  if (idx >= len) idx = len - 1;
  return idx;
}

}  // namespace

struct RmvpeF0::Impl {
  Ort::Session* sess = nullptr;
  std::string in_name;
  std::string out_name;

  // 形状约定（基于运行时探测）：
  // - 输入： [1, 128, T] 或 [1, T, 128]
  // - 输出： [1, T, 360] 或 [1, 360, T]（或去掉 batch）
  bool input_bct = true;     // true=[1,128,T]，false=[1,T,128]
  bool output_t360 = true;   // true=[T,360]，false=[360,T]

  // 前处理常量（对齐 infer/lib/rmvpe.py）
  int32_t sr = 16000;
  int32_t n_fft = 1024;
  int32_t hop = 160;
  int32_t win_length = 1024;
  int32_t n_mels = 128;
  float mel_fmin = 30.0f;
  float mel_fmax = 8000.0f;
  float clamp = 1e-5f;

  // 预计算
  std::vector<float> hann;       // [win_length]
  std::vector<float> mel_basis;  // [n_mels, n_freq]
  std::vector<float> cents_map;  // [368] = pad(20*arange(360)+1997.3794, 4)

  kiss_fftr_cfg fft_cfg = nullptr;
};

RmvpeF0::RmvpeF0() : impl_(std::make_unique<Impl>()) {}

RmvpeF0::~RmvpeF0() {
  if (impl_ && impl_->fft_cfg) {
    kiss_fftr_free(impl_->fft_cfg);
    impl_->fft_cfg = nullptr;
  }
}

bool RmvpeF0::Init(Ort::Session* sess, Error* err) {
  if (!impl_) return false;
  if (!sess) {
    SetErr_(err, 200, "rmvpe session is null.");
    return false;
  }
  impl_->sess = sess;

  // IO 名称
  try {
    Ort::AllocatorWithDefaultOptions alloc;
    if (sess->GetInputCount() < 1 || sess->GetOutputCount() < 1) {
      SetErr_(err, 201, "rmvpe.onnx must have at least 1 input and 1 output.");
      return false;
    }
    auto in0 = sess->GetInputNameAllocated(0, alloc);
    auto out0 = sess->GetOutputNameAllocated(0, alloc);
    impl_->in_name = in0.get();
    impl_->out_name = out0.get();

    // 探测输入维度：优先识别 128 所在的维度
    auto ti = sess->GetInputTypeInfo(0).GetTensorTypeAndShapeInfo();
    std::vector<int64_t> in_shape = ti.GetShape();
    if (in_shape.size() == 3 && in_shape[0] == 1) {
      // [1,128,T] 或 [1,T,128]
      if (in_shape[1] == impl_->n_mels) {
        impl_->input_bct = true;
      } else if (in_shape[2] == impl_->n_mels) {
        impl_->input_bct = false;
      } else {
        // 形状未知时默认 [1,128,T]，运行时失败会抛出 ORT 异常
        impl_->input_bct = true;
      }
    }
  } catch (const Ort::Exception& e) {
    SetErr_(err, 202, std::string("rmvpe init failed: ") + e.what());
    return false;
  } catch (...) {
    SetErr_(err, 203, "rmvpe init failed: unknown exception.");
    return false;
  }

  // 预计算 window/mel/cents
  BuildHannWindow_(impl_->win_length, &impl_->hann);
  BuildMelBasisHTK_(impl_->sr, impl_->n_fft, impl_->n_mels, impl_->mel_fmin, impl_->mel_fmax, &impl_->mel_basis);

  impl_->cents_map.assign(368, 0.0f);
  const double base = 1997.3794084376191;  // 对齐 python 常量
  for (int i = 0; i < 360; ++i) {
    impl_->cents_map[static_cast<size_t>(i + 4)] = static_cast<float>(20.0 * i + base);
  }

  // FFT cfg
  if (impl_->fft_cfg) {
    kiss_fftr_free(impl_->fft_cfg);
    impl_->fft_cfg = nullptr;
  }
  impl_->fft_cfg = kiss_fftr_alloc(impl_->n_fft, 0, nullptr, nullptr);
  if (!impl_->fft_cfg) {
    SetErr_(err, 204, "rmvpe init failed: kiss_fftr_alloc returned null.");
    return false;
  }

  return true;
}

bool RmvpeF0::ComputeF0Hz(const float* audio16k,
                          int32_t audio_len,
                          float thred,
                          std::vector<float>* out_f0_hz,
                          Error* err) const {
  if (!impl_ || !impl_->sess || !out_f0_hz) return false;
  out_f0_hz->clear();
  if (!audio16k || audio_len <= 0) return true;

  // 10ms hop 对齐：输出长度与 python 的习惯一致：len//hop + 1
  const int32_t hop = impl_->hop;
  const int32_t n_frames = audio_len / hop + 1;
  if (n_frames <= 0) return true;

  // center=True 的 reflect padding（pad = n_fft/2）
  const int32_t pad = impl_->n_fft / 2;
  const int32_t padded_len = audio_len + 2 * pad;
  std::vector<float> padded(static_cast<size_t>(padded_len));
  for (int32_t i = 0; i < padded_len; ++i) {
    const int32_t idx = ReflectIndex_(i - pad, audio_len);
    padded[static_cast<size_t>(i)] = audio16k[idx];
  }

  // rmvpe 模型里通常要求 time 维度对齐到 32 的倍数（python 内部会 pad mel）
  const int32_t n_pad = 32 * ((n_frames - 1) / 32 + 1) - n_frames;
  const int32_t frames_pad = (n_pad > 0) ? (n_frames + n_pad) : n_frames;

  // 生成 log-mel（float32）
  const int32_t n_freq = impl_->n_fft / 2 + 1;
  std::vector<float> mel_in;
  if (impl_->input_bct) {
    mel_in.assign(static_cast<size_t>(impl_->n_mels) * static_cast<size_t>(frames_pad), 0.0f);
  } else {
    mel_in.assign(static_cast<size_t>(frames_pad) * static_cast<size_t>(impl_->n_mels), 0.0f);
  }

  std::vector<kiss_fft_scalar> frame(static_cast<size_t>(impl_->n_fft));
  std::vector<kiss_fft_cpx> freq(static_cast<size_t>(n_freq));
  std::vector<float> mag(static_cast<size_t>(n_freq));

  for (int32_t t = 0; t < n_frames; ++t) {
    const int32_t start = t * hop;
    // padded 长度保证 start+ n_fft 可取（因为 padded_len = audio_len + n_fft）
    const float* x = padded.data() + start;

    // 加窗
    for (int32_t i = 0; i < impl_->n_fft; ++i) {
      const float w = (i < static_cast<int32_t>(impl_->hann.size())) ? impl_->hann[static_cast<size_t>(i)] : 1.0f;
      frame[static_cast<size_t>(i)] = static_cast<kiss_fft_scalar>(x[i] * w);
    }

    // rfft
    kiss_fftr(impl_->fft_cfg, frame.data(), freq.data());
    for (int32_t k = 0; k < n_freq; ++k) {
      const float re = static_cast<float>(freq[static_cast<size_t>(k)].r);
      const float im = static_cast<float>(freq[static_cast<size_t>(k)].i);
      mag[static_cast<size_t>(k)] = std::sqrt(re * re + im * im);
    }

    // mel + log
    for (int32_t m = 0; m < impl_->n_mels; ++m) {
      const float* basis = impl_->mel_basis.data() + static_cast<size_t>(m) * static_cast<size_t>(n_freq);
      double s = 0.0;
      for (int32_t k = 0; k < n_freq; ++k) {
        s += (double)basis[static_cast<size_t>(k)] * (double)mag[static_cast<size_t>(k)];
      }
      float v = static_cast<float>(s);
      if (v < impl_->clamp) v = impl_->clamp;
      v = std::log(v);

      if (impl_->input_bct) {
        // [128, T]
        mel_in[static_cast<size_t>(m) * static_cast<size_t>(frames_pad) + static_cast<size_t>(t)] = v;
      } else {
        // [T, 128]
        mel_in[static_cast<size_t>(t) * static_cast<size_t>(impl_->n_mels) + static_cast<size_t>(m)] = v;
      }
    }
  }
  // pad 的部分保持 0（与 python 的 constant pad 对齐）

  // ORT run
  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::vector<int64_t> in_shape;
  if (impl_->input_bct) {
    in_shape = {1, impl_->n_mels, frames_pad};
  } else {
    in_shape = {1, frames_pad, impl_->n_mels};
  }

  Ort::Value in_t = Ort::Value::CreateTensor<float>(
      mem, mel_in.data(), mel_in.size(), in_shape.data(), in_shape.size());

  const char* in_names[] = {impl_->in_name.c_str()};
  const char* out_names[] = {impl_->out_name.c_str()};

  std::vector<Ort::Value> out;
  try {
    out = impl_->sess->Run(Ort::RunOptions{nullptr}, in_names, &in_t, 1, out_names, 1);
  } catch (const Ort::Exception& e) {
    SetErr_(err, 210, std::string("rmvpe run failed: ") + e.what());
    return false;
  }
  if (out.empty()) {
    SetErr_(err, 211, "rmvpe produced no outputs.");
    return false;
  }

  auto info = out[0].GetTensorTypeAndShapeInfo();
  std::vector<int64_t> shape = info.GetShape();
  const float* data = out[0].GetTensorData<float>();
  if (!data) {
    SetErr_(err, 212, "rmvpe output data is null.");
    return false;
  }

  // 去掉 batch=1
  if (shape.size() == 3 && shape[0] == 1) {
    shape.erase(shape.begin());
  }
  if (shape.size() != 2) {
    SetErr_(err, 213, "rmvpe output must be 2D (after removing batch).");
    return false;
  }

  const int64_t a = shape[0];
  const int64_t b = shape[1];
  int32_t out_frames = 0;
  bool t360 = true;
  if (b == 360) {
    out_frames = static_cast<int32_t>(a);
    t360 = true;   // [T,360]
  } else if (a == 360) {
    out_frames = static_cast<int32_t>(b);
    t360 = false;  // [360,T]
  } else {
    SetErr_(err, 214, "rmvpe output dims do not contain 360.");
    return false;
  }
  if (out_frames <= 0) {
    SetErr_(err, 215, "rmvpe output T is invalid.");
    return false;
  }

  // 解码
  out_f0_hz->assign(static_cast<size_t>(n_frames), 0.0f);
  const float thr = (thred >= 0.0f) ? thred : 0.0f;

  auto get_sal = [&](int32_t t, int32_t k) -> float {
    // t: [0,out_frames), k:[0,360)
    if (t < 0 || t >= out_frames || k < 0 || k >= 360) return 0.0f;
    if (t360) {
      return data[static_cast<size_t>(t) * 360 + static_cast<size_t>(k)];
    }
    // [360,T]：k 为行，t 为列
    return data[static_cast<size_t>(k) * static_cast<size_t>(out_frames) + static_cast<size_t>(t)];
  };

  const int32_t frames_use = std::min<int32_t>(n_frames, out_frames);
  for (int32_t t = 0; t < frames_use; ++t) {
    int32_t center = 0;
    float maxv = -1e30f;
    for (int32_t k = 0; k < 360; ++k) {
      const float v = get_sal(t, k);
      if (v > maxv) {
        maxv = v;
        center = k;
      }
    }
    if (maxv <= thr) {
      (*out_f0_hz)[static_cast<size_t>(t)] = 0.0f;
      continue;
    }

    // local average cents（对齐 python：center 周围 9 个 bin）
    double sum = 0.0;
    double wsum = 0.0;
    for (int32_t j = -4; j <= 4; ++j) {
      const int32_t kk = center + j;
      const float s = (kk >= 0 && kk < 360) ? get_sal(t, kk) : 0.0f;
      const float cents = impl_->cents_map[static_cast<size_t>(kk + 4)];
      sum += (double)s * (double)cents;
      wsum += (double)s;
    }
    if (wsum <= 0.0) {
      (*out_f0_hz)[static_cast<size_t>(t)] = 0.0f;
      continue;
    }
    const double cents_pred = sum / wsum;
    if (cents_pred <= 0.0) {
      (*out_f0_hz)[static_cast<size_t>(t)] = 0.0f;
      continue;
    }
    const float f0 = 10.0f * std::pow(2.0f, static_cast<float>(cents_pred / 1200.0));
    (*out_f0_hz)[static_cast<size_t>(t)] = f0;
  }

  return true;
}

}  // namespace rvc_ort

