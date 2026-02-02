// yin_f0.h
// YIN 音高提取（简化实现）：用于在运行时无 Python 的情况下提供可用的 F0。
// 说明：
// - 输入采样率固定 16k（与本仓库实时链路对齐）
// - hop 固定 160（10ms）
// - 输出长度为 p_len（10ms 帧数）

#pragma once

#include <cstdint>
#include <vector>

namespace rvc_ort {

// 计算给定音频片段的 f0（Hz）。
// - audio: 16k 单声道
// - sr: 16000
// - hop: 160（10ms）
// - f0_min/f0_max: 频率范围
// - out_f0: 输出（长度约为 audio.size()/hop + 1）
//
// 返回：out_f0 的长度
int32_t ComputeF0Yin16k(const float* audio,
                        int32_t audio_len,
                        int32_t sr,
                        int32_t hop,
                        float f0_min,
                        float f0_max,
                        std::vector<float>* out_f0);

}  // namespace rvc_ort

