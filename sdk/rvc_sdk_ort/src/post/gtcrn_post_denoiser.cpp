#include "post/gtcrn_post_denoiser.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#ifdef _WIN32
#include <Windows.h>
#endif

extern "C" {
#include "kiss_fftr.h"
}

#include "dsp/linear_resampler.h"

namespace rvc_ort {
namespace {

constexpr int32_t kGtcrnSr = 16000;
constexpr int32_t kFft = 512;
constexpr int32_t kHop = 256;
constexpr int32_t kBins = kFft / 2 + 1;

constexpr int32_t kConvN = 2 * 1 * 16 * 16 * 33;
constexpr int32_t kTraN = 2 * 3 * 1 * 1 * 16;
constexpr int32_t kInterN = 2 * 1 * 33 * 16;

static void SetErr(std::string* err, const std::string& msg) {
  if (!err) return;
  *err = msg;
}

#ifdef _WIN32
static std::wstring Utf8OrAcpToWide(const std::string& s) {
  if (s.empty()) return std::wstring();

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

static int32_t CalcResampledLen(int32_t in_len, int32_t in_sr, int32_t out_sr) {
  if (in_len <= 0 || in_sr <= 0 || out_sr <= 0) return 0;
  if (in_sr == out_sr) return in_len;
  const int64_t v = static_cast<int64_t>(in_len) * static_cast<int64_t>(out_sr);
  return static_cast<int32_t>((v + in_sr / 2) / in_sr);
}

}  // namespace

struct GtcrnPostDenoiser::KissFftWrap {
  kiss_fftr_cfg cfg_fwd = nullptr;
  kiss_fftr_cfg cfg_inv = nullptr;
  std::vector<kiss_fft_cpx> freq;

  ~KissFftWrap() {
    if (cfg_fwd) {
      kiss_fftr_free(cfg_fwd);
      cfg_fwd = nullptr;
    }
    if (cfg_inv) {
      kiss_fftr_free(cfg_inv);
      cfg_inv = nullptr;
    }
  }
};

GtcrnPostDenoiser::GtcrnPostDenoiser() = default;

GtcrnPostDenoiser::~GtcrnPostDenoiser() = default;

bool GtcrnPostDenoiser::Init(Ort::Env* env,
                             const Ort::SessionOptions& so,
                             const std::string& onnx_path,
                             std::string* err) {
  if (!env) {
    SetErr(err, "gtcrn init: Ort::Env is null.");
    return false;
  }
  if (onnx_path.empty()) {
    SetErr(err, "gtcrn init: onnx path is empty.");
    return false;
  }

  try {
#ifdef _WIN32
    const std::wstring model_w = Utf8OrAcpToWide(onnx_path);
    sess_ = std::make_unique<Ort::Session>(*env, model_w.c_str(), so);
#else
    sess_ = std::make_unique<Ort::Session>(*env, onnx_path.c_str(), so);
#endif
  } catch (const Ort::Exception& e) {
    SetErr(err, std::string("gtcrn init: failed to create ORT session: ") + e.what());
    return false;
  }

  try {
    Ort::AllocatorWithDefaultOptions alloc;
    const size_t ni = sess_->GetInputCount();
    const size_t no = sess_->GetOutputCount();
    if (ni < 4 || no < 4) {
      SetErr(err, "gtcrn init: model must have at least 4 inputs and 4 outputs.");
      return false;
    }

    in_names_.clear();
    out_names_.clear();
    in_names_.reserve(4);
    out_names_.reserve(4);
    for (size_t i = 0; i < 4; ++i) {
      auto name = sess_->GetInputNameAllocated(i, alloc);
      in_names_.push_back(name.get());
    }
    for (size_t i = 0; i < 4; ++i) {
      auto name = sess_->GetOutputNameAllocated(i, alloc);
      out_names_.push_back(name.get());
    }
  } catch (const Ort::Exception& e) {
    SetErr(err, std::string("gtcrn init: failed to read io names: ") + e.what());
    return false;
  }

  conv_cache_.assign(kConvN, 0.0f);
  tra_cache_.assign(kTraN, 0.0f);
  inter_cache_.assign(kInterN, 0.0f);

  window_.assign(kFft, 0.0f);
  constexpr float kPi = 3.14159265358979323846f;
  for (int32_t i = 0; i < kFft; ++i) {
    const float w = 0.5f - 0.5f * std::cos(2.0f * kPi * static_cast<float>(i) / static_cast<float>(kFft));
    window_[static_cast<size_t>(i)] = std::sqrt(std::max(w, 0.0f));
  }

  input_cache_.assign(kFft, 0.0f);
  output_cache_.assign(kFft, 0.0f);
  pending_in16_.clear();
  pending_out16_.clear();

  fft_in_.assign(kFft, 0.0f);
  ifft_out_.assign(kFft, 0.0f);
  mix_ri_.assign(kBins * 2, 0.0f);
  hop_out_.assign(kHop, 0.0f);

  fft_ = std::make_unique<KissFftWrap>();
  fft_->freq.assign(kBins, kiss_fft_cpx{});
  fft_->cfg_fwd = kiss_fftr_alloc(kFft, 0, nullptr, nullptr);
  fft_->cfg_inv = kiss_fftr_alloc(kFft, 1, nullptr, nullptr);
  if (!fft_->cfg_fwd || !fft_->cfg_inv) {
    SetErr(err, "gtcrn init: failed to initialize FFT state.");
    return false;
  }

  return true;
}

void GtcrnPostDenoiser::Reset() {
  std::fill(conv_cache_.begin(), conv_cache_.end(), 0.0f);
  std::fill(tra_cache_.begin(), tra_cache_.end(), 0.0f);
  std::fill(inter_cache_.begin(), inter_cache_.end(), 0.0f);
  std::fill(input_cache_.begin(), input_cache_.end(), 0.0f);
  std::fill(output_cache_.begin(), output_cache_.end(), 0.0f);
  pending_in16_.clear();
  pending_out16_.clear();
}

bool GtcrnPostDenoiser::ProcessHop16_(const float* hop_in, float* hop_out, std::string* err) {
  if (!sess_ || !fft_ || !hop_in || !hop_out) {
    SetErr(err, "gtcrn process: invalid state.");
    return false;
  }

  std::memmove(input_cache_.data(), input_cache_.data() + kHop, sizeof(float) * (kFft - kHop));
  std::memcpy(input_cache_.data() + (kFft - kHop), hop_in, sizeof(float) * kHop);

  for (int32_t i = 0; i < kFft; ++i) {
    fft_in_[static_cast<size_t>(i)] = input_cache_[static_cast<size_t>(i)] * window_[static_cast<size_t>(i)];
  }

  kiss_fftr(fft_->cfg_fwd, fft_in_.data(), fft_->freq.data());
  for (int32_t k = 0; k < kBins; ++k) {
    mix_ri_[static_cast<size_t>(2 * k)] = fft_->freq[static_cast<size_t>(k)].r;
    mix_ri_[static_cast<size_t>(2 * k + 1)] = fft_->freq[static_cast<size_t>(k)].i;
  }

  Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
  std::array<int64_t, 4> mix_shape{1, kBins, 1, 2};
  std::array<int64_t, 5> conv_shape{2, 1, 16, 16, 33};
  std::array<int64_t, 5> tra_shape{2, 3, 1, 1, 16};
  std::array<int64_t, 4> inter_shape{2, 1, 33, 16};

  std::array<Ort::Value, 4> inputs{
      Ort::Value::CreateTensor<float>(mem, mix_ri_.data(), mix_ri_.size(), mix_shape.data(), mix_shape.size()),
      Ort::Value::CreateTensor<float>(mem, conv_cache_.data(), conv_cache_.size(), conv_shape.data(), conv_shape.size()),
      Ort::Value::CreateTensor<float>(mem, tra_cache_.data(), tra_cache_.size(), tra_shape.data(), tra_shape.size()),
      Ort::Value::CreateTensor<float>(mem, inter_cache_.data(), inter_cache_.size(), inter_shape.data(), inter_shape.size())};

  const char* in_names[4] = {
      in_names_[0].c_str(),
      in_names_[1].c_str(),
      in_names_[2].c_str(),
      in_names_[3].c_str()};
  const char* out_names[4] = {
      out_names_[0].c_str(),
      out_names_[1].c_str(),
      out_names_[2].c_str(),
      out_names_[3].c_str()};

  std::vector<Ort::Value> out;
  try {
    out = sess_->Run(Ort::RunOptions{nullptr}, in_names, inputs.data(), 4, out_names, 4);
  } catch (const Ort::Exception& e) {
    SetErr(err, std::string("gtcrn process: ORT run failed: ") + e.what());
    return false;
  }

  if (out.size() < 4) {
    SetErr(err, "gtcrn process: model outputs are incomplete.");
    return false;
  }

  const float* enh = out[0].GetTensorData<float>();
  const size_t enh_n = out[0].GetTensorTypeAndShapeInfo().GetElementCount();
  if (!enh || enh_n < static_cast<size_t>(kBins * 2)) {
    SetErr(err, "gtcrn process: invalid enhanced spectrum shape.");
    return false;
  }

  const float* conv = out[1].GetTensorData<float>();
  const float* tra = out[2].GetTensorData<float>();
  const float* inter = out[3].GetTensorData<float>();
  const size_t conv_n = out[1].GetTensorTypeAndShapeInfo().GetElementCount();
  const size_t tra_n = out[2].GetTensorTypeAndShapeInfo().GetElementCount();
  const size_t inter_n = out[3].GetTensorTypeAndShapeInfo().GetElementCount();
  if (!conv || !tra || !inter ||
      conv_n != conv_cache_.size() ||
      tra_n != tra_cache_.size() ||
      inter_n != inter_cache_.size()) {
    SetErr(err, "gtcrn process: invalid cache tensor shape.");
    return false;
  }

  std::memcpy(conv_cache_.data(), conv, sizeof(float) * conv_cache_.size());
  std::memcpy(tra_cache_.data(), tra, sizeof(float) * tra_cache_.size());
  std::memcpy(inter_cache_.data(), inter, sizeof(float) * inter_cache_.size());

  for (int32_t k = 0; k < kBins; ++k) {
    fft_->freq[static_cast<size_t>(k)].r = enh[static_cast<size_t>(2 * k)];
    fft_->freq[static_cast<size_t>(k)].i = enh[static_cast<size_t>(2 * k + 1)];
  }
  kiss_fftri(fft_->cfg_inv, fft_->freq.data(), ifft_out_.data());

  constexpr float kInvFft = 1.0f / static_cast<float>(kFft);
  for (int32_t i = 0; i < kFft; ++i) {
    const float frame = ifft_out_[static_cast<size_t>(i)] * kInvFft * window_[static_cast<size_t>(i)];
    output_cache_[static_cast<size_t>(i)] += frame;
  }

  std::memcpy(hop_out, output_cache_.data(), sizeof(float) * kHop);
  std::memmove(output_cache_.data(), output_cache_.data() + kHop, sizeof(float) * (kFft - kHop));
  std::fill(output_cache_.begin() + (kFft - kHop), output_cache_.end(), 0.0f);
  return true;
}

bool GtcrnPostDenoiser::ProcessBlock(const float* in_io,
                                     int32_t in_len,
                                     int32_t io_sr,
                                     float* out_io,
                                     std::string* err) {
  if (!in_io || !out_io || in_len <= 0) {
    SetErr(err, "gtcrn process: invalid input/output block.");
    return false;
  }
  if (io_sr <= 0) {
    SetErr(err, "gtcrn process: invalid io_sr.");
    return false;
  }

  const int32_t out_len16 = CalcResampledLen(in_len, io_sr, kGtcrnSr);
  if (out_len16 <= 0) {
    SetErr(err, "gtcrn process: invalid resampled length.");
    return false;
  }

  std::vector<float> block_in16(static_cast<size_t>(out_len16), 0.0f);
  if (io_sr == kGtcrnSr) {
    std::memcpy(block_in16.data(), in_io, sizeof(float) * static_cast<size_t>(out_len16));
  } else {
    ResampleLinear(in_io, in_len, block_in16.data(), out_len16);
  }

  pending_in16_.insert(pending_in16_.end(), block_in16.begin(), block_in16.end());

  size_t consume = 0;
  while (pending_in16_.size() - consume >= static_cast<size_t>(kHop)) {
    if (!ProcessHop16_(pending_in16_.data() + consume, hop_out_.data(), err)) {
      return false;
    }
    pending_out16_.insert(pending_out16_.end(), hop_out_.begin(), hop_out_.end());
    consume += static_cast<size_t>(kHop);
  }
  if (consume > 0) {
    pending_in16_.erase(pending_in16_.begin(), pending_in16_.begin() + static_cast<int64_t>(consume));
  }

  std::vector<float> block_out16(static_cast<size_t>(out_len16), 0.0f);
  const size_t avail = std::min(block_out16.size(), pending_out16_.size());
  if (avail > 0) {
    std::memcpy(block_out16.data(), pending_out16_.data(), sizeof(float) * avail);
    pending_out16_.erase(pending_out16_.begin(), pending_out16_.begin() + static_cast<int64_t>(avail));
  }
  if (avail < block_out16.size()) {
    std::memcpy(block_out16.data() + avail,
                block_in16.data() + avail,
                sizeof(float) * (block_out16.size() - avail));
  }

  if (io_sr == kGtcrnSr && out_len16 == in_len) {
    std::memcpy(out_io, block_out16.data(), sizeof(float) * static_cast<size_t>(in_len));
  } else {
    ResampleLinear(block_out16.data(), out_len16, out_io, in_len);
  }
  return true;
}

}  // namespace rvc_ort
