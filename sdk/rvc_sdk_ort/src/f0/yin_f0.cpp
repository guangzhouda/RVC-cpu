// yin_f0.cpp

#include "f0/yin_f0.h"

#include <algorithm>
#include <cmath>

namespace rvc_ort {

static inline int32_t ClampI32(int32_t v, int32_t lo, int32_t hi) {
  return std::max(lo, std::min(hi, v));
}

static float ParabolicInterp(float s0, float s1, float s2) {
  // 顶点偏移（-0.5~0.5），用于对 tau 做亚采样修正
  const float denom = (s0 - 2.0f * s1 + s2);
  if (std::abs(denom) < 1e-12f) {
    return 0.0f;
  }
  return 0.5f * (s0 - s2) / denom;
}

int32_t ComputeF0Yin16k(const float* audio,
                        int32_t audio_len,
                        int32_t sr,
                        int32_t hop,
                        float f0_min,
                        float f0_max,
                        std::vector<float>* out_f0) {
  if (!out_f0) {
    return 0;
  }
  out_f0->clear();
  if (!audio || audio_len <= 0 || sr <= 0 || hop <= 0) {
    return 0;
  }
  if (f0_min <= 0.0f || f0_max <= f0_min) {
    return 0;
  }

  // YIN 参数：窗长与阈值
  // 为了实时性，这里选用固定 frame_size；并仅计算 tau_min..tau_max
  const int32_t frame_size = 1024;  // 约 64ms（>= 2*tau_max 的要求一般更稳）
  const float yin_thresh = 0.10f;   // 经验阈值（越小越严格）

  const int32_t tau_min = std::max<int32_t>(2, static_cast<int32_t>(std::floor(sr / f0_max)));
  const int32_t tau_max = std::max<int32_t>(tau_min + 1, static_cast<int32_t>(std::ceil(sr / f0_min)));
  const int32_t max_tau = std::min<int32_t>(tau_max, frame_size / 2);

  // 输出帧数（与本仓库习惯对齐：len//hop + 1）
  const int32_t p_len = audio_len / hop + 1;
  out_f0->resize(p_len, 0.0f);

  std::vector<float> frame(frame_size);
  std::vector<float> diff(max_tau + 1);
  std::vector<float> cmndf(max_tau + 1);

  for (int32_t fi = 0; fi < p_len; ++fi) {
    const int32_t center = fi * hop;

    // 取 frame_size 的窗口，使用 reflect padding（接近 librosa/pad 的效果）
    const int32_t half = frame_size / 2;
    for (int32_t i = 0; i < frame_size; ++i) {
      int32_t idx = center - half + i;
      if (idx < 0) idx = -idx;  // reflect
      if (idx >= audio_len) idx = audio_len - 1 - (idx - (audio_len - 1));  // reflect
      idx = ClampI32(idx, 0, audio_len - 1);
      frame[i] = audio[idx];
    }

    // difference function
    std::fill(diff.begin(), diff.end(), 0.0f);
    for (int32_t tau = 1; tau <= max_tau; ++tau) {
      float s = 0.0f;
      for (int32_t j = 0; j < frame_size - tau; ++j) {
        const float d = frame[j] - frame[j + tau];
        s += d * d;
      }
      diff[tau] = s;
    }

    // cumulative mean normalized difference function
    cmndf[0] = 1.0f;
    float running_sum = 0.0f;
    for (int32_t tau = 1; tau <= max_tau; ++tau) {
      running_sum += diff[tau];
      cmndf[tau] = (running_sum > 0.0f) ? (diff[tau] * tau / running_sum) : 1.0f;
    }

    // 找 tau：先阈值，再找局部最小
    int32_t tau_est = 0;
    for (int32_t tau = tau_min; tau <= max_tau; ++tau) {
      if (cmndf[tau] < yin_thresh) {
        tau_est = tau;
        while (tau_est + 1 <= max_tau && cmndf[tau_est + 1] < cmndf[tau_est]) {
          ++tau_est;
        }
        break;
      }
    }
    if (tau_est == 0) {
      (*out_f0)[fi] = 0.0f;
      continue;
    }

    // 抛物线插值（提升频率精度）
    const int32_t t0 = std::max<int32_t>(tau_est - 1, 1);
    const int32_t t1 = tau_est;
    const int32_t t2 = std::min<int32_t>(tau_est + 1, max_tau);
    const float shift = ParabolicInterp(cmndf[t0], cmndf[t1], cmndf[t2]);
    const float tau_refined = static_cast<float>(tau_est) + shift;
    const float f0 = (tau_refined > 1e-6f) ? (static_cast<float>(sr) / tau_refined) : 0.0f;

    if (f0 < f0_min || f0 > f0_max) {
      (*out_f0)[fi] = 0.0f;
    } else {
      (*out_f0)[fi] = f0;
    }
  }

  return static_cast<int32_t>(out_f0->size());
}

}  // namespace rvc_ort

