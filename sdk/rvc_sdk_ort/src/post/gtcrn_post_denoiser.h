#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace rvc_ort {

class GtcrnPostDenoiser {
 public:
  GtcrnPostDenoiser();
  ~GtcrnPostDenoiser();

  bool Init(Ort::Env* env,
            const Ort::SessionOptions& so,
            const std::string& onnx_path,
            std::string* err);

  void Reset();

  bool ProcessBlock(const float* in_io,
                    int32_t in_len,
                    int32_t io_sr,
                    float* out_io,
                    std::string* err);

 private:
  bool ProcessHop16_(const float* hop_in, float* hop_out, std::string* err);

 private:
  std::unique_ptr<Ort::Session> sess_;

  std::vector<std::string> in_names_;
  std::vector<std::string> out_names_;

  std::vector<float> conv_cache_;
  std::vector<float> tra_cache_;
  std::vector<float> inter_cache_;

  std::vector<float> window_;
  std::vector<float> input_cache_;
  std::vector<float> output_cache_;

  std::vector<float> pending_in16_;
  std::vector<float> pending_out16_;

  std::vector<float> fft_in_;
  std::vector<float> ifft_out_;
  std::vector<float> mix_ri_;
  std::vector<float> hop_out_;

  struct KissFftWrap;
  std::unique_ptr<KissFftWrap> fft_;
};

}  // namespace rvc_ort
