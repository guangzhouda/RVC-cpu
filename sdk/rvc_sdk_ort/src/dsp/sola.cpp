// sola.cpp

#include "dsp/sola.h"

#include <algorithm>
#include <cmath>

namespace rvc_ort {

static float Dot(const float* a, const float* b, int32_t n) {
  float s = 0.0f;
  for (int32_t i = 0; i < n; ++i) {
    s += a[i] * b[i];
  }
  return s;
}

static float Energy(const float* a, int32_t n) {
  float s = 0.0f;
  for (int32_t i = 0; i < n; ++i) {
    s += a[i] * a[i];
  }
  return s;
}

Sola::Sola(const SolaConfig& cfg) : cfg_(cfg) {
  sola_buffer_.assign(std::max<int32_t>(0, cfg_.sola_buffer_size), 0.0f);
}

void Sola::Reset() {
  std::fill(sola_buffer_.begin(), sola_buffer_.end(), 0.0f);
}

int32_t Sola::Process(const float* infer_wav,
                      int32_t infer_len,
                      int32_t block_size,
                      const float* fade_in,
                      const float* fade_out,
                      float* out_block) {
  if (!infer_wav || !out_block || block_size <= 0) {
    return 0;
  }
  const int32_t B = cfg_.sola_buffer_size;
  const int32_t S = cfg_.sola_search_size;
  if (B <= 0 || S < 0) {
    // 无对齐：直接拷贝 block
    std::copy(infer_wav, infer_wav + std::min<int32_t>(block_size, infer_len), out_block);
    return 0;
  }
  const int32_t need = B + S + block_size;
  if (infer_len < need) {
    // 输入不足时，尽力输出（避免越界）
    std::fill(out_block, out_block + block_size, 0.0f);
    const int32_t copy_n = std::min<int32_t>(block_size, infer_len);
    std::copy(infer_wav, infer_wav + copy_n, out_block);
    return 0;
  }

  // 在 infer_wav[0 : B+S] 中寻找与 sola_buffer 最匹配的偏移
  // 与 python 版等价：argmax( cor_nom / sqrt(cor_den + eps) )
  const float eps = 1e-8f;

  int32_t best_off = 0;
  float best_score = -1e30f;
  const float sola_energy = std::max<float>(Energy(sola_buffer_.data(), B), eps);

  for (int32_t off = 0; off <= S; ++off) {
    const float* x = infer_wav + off;
    const float nom = Dot(x, sola_buffer_.data(), B);
    const float den = std::sqrt(std::max<float>(Energy(x, B), eps) * sola_energy);
    const float score = nom / den;
    if (score > best_score) {
      best_score = score;
      best_off = off;
    }
  }

  const float* aligned = infer_wav + best_off;

  // crossfade（前 B 个采样点）
  if (fade_in && fade_out && static_cast<int32_t>(sola_buffer_.size()) == B) {
    for (int32_t i = 0; i < B; ++i) {
      const float a = aligned[i] * fade_in[i];
      const float b = sola_buffer_[i] * fade_out[i];
      // 与 python 的实现对齐：infer_wav[:B] = infer_wav[:B]*fade_in + sola_buffer*fade_out
      // 这里直接写入 aligned 前缀到临时输出中。
      out_block[i] = a + b;
    }
    // block 剩余部分直接拷贝
    std::copy(aligned + B, aligned + block_size, out_block + B);
  } else {
    std::copy(aligned, aligned + block_size, out_block);
  }

  // 更新 sola_buffer：取 aligned 的 [block_size : block_size + B]
  if (static_cast<int32_t>(sola_buffer_.size()) == B) {
    std::copy(aligned + block_size, aligned + block_size + B, sola_buffer_.begin());
  }

  return best_off;
}

}  // namespace rvc_ort

