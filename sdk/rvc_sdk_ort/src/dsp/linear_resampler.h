// linear_resampler.h
// 说明：为了减少第三方依赖，这里先实现一个简单线性重采样器。
// 后续如需更高音质/更低延迟，可替换为 soxr 等成熟库。

#pragma once

#include <cstdint>
#include <vector>

namespace rvc_ort {

// 将输入 in（in_len）线性重采样到 out_len。
// 注意：这是“块内重采样”，不维护跨块滤波状态；实时场景下可能在极端条件产生轻微边界误差。
void ResampleLinear(const float* in, int32_t in_len, float* out, int32_t out_len);

// 便捷版本：返回 vector
std::vector<float> ResampleLinear(const std::vector<float>& in, int32_t out_len);

}  // namespace rvc_ort

