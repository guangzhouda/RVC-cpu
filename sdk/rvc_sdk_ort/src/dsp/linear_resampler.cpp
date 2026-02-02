// linear_resampler.cpp

#include "dsp/linear_resampler.h"

#include <algorithm>
#include <cmath>

namespace rvc_ort {

void ResampleLinear(const float* in, int32_t in_len, float* out, int32_t out_len) {
  if (!in || !out || in_len <= 0 || out_len <= 0) {
    return;
  }
  if (in_len == 1) {
    std::fill(out, out + out_len, in[0]);
    return;
  }
  if (out_len == 1) {
    out[0] = in[0];
    return;
  }

  // 以“端点对齐”的方式映射：out[0] -> in[0]，out[last] -> in[last]
  const double scale = static_cast<double>(in_len - 1) / static_cast<double>(out_len - 1);

  for (int32_t i = 0; i < out_len; ++i) {
    const double x = scale * static_cast<double>(i);
    const int32_t x0 = static_cast<int32_t>(std::floor(x));
    const int32_t x1 = std::min<int32_t>(x0 + 1, in_len - 1);
    const float t = static_cast<float>(x - static_cast<double>(x0));
    out[i] = (1.0f - t) * in[x0] + t * in[x1];
  }
}

std::vector<float> ResampleLinear(const std::vector<float>& in, int32_t out_len) {
  std::vector<float> out;
  out.resize(std::max<int32_t>(0, out_len));
  if (!out.empty()) {
    ResampleLinear(in.data(), static_cast<int32_t>(in.size()), out.data(), out_len);
  }
  return out;
}

}  // namespace rvc_ort

