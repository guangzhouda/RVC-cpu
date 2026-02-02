// sola.h
// SOLA（Synchronous OverLap-Add）对齐：用于降低 block 边界伪影。
// 参考实现逻辑与本仓库 `api_240604.py` 保持一致（相关性搜索 + crossfade）。

#pragma once

#include <cstdint>
#include <vector>

namespace rvc_ort {

struct SolaConfig {
  int32_t sola_buffer_size = 0;  // crossfade 缓冲长度（采样点）
  int32_t sola_search_size = 0;  // 搜索范围（采样点）
};

class Sola {
 public:
  explicit Sola(const SolaConfig& cfg);

  void Reset();

  // 输入 infer_wav（至少包含 sola_buffer_size + sola_search_size + block_size 的数据）
  // 输出：对齐后的 block（长度 block_size），并更新内部 sola_buffer。
  //
  // - fade_in/out：长度 sola_buffer_size 的窗函数
  // - 返回值：sola_offset（0~sola_search_size），便于调试
  int32_t Process(const float* infer_wav,
                  int32_t infer_len,
                  int32_t block_size,
                  const float* fade_in,
                  const float* fade_out,
                  float* out_block);

 private:
  SolaConfig cfg_;
  std::vector<float> sola_buffer_;
};

}  // namespace rvc_ort

