// rmvpe_f0.h
// RMVPE 音高提取（ORT ONNX）：运行时无 Python。
//
// 说明：
// - RMVPE 模型（rmvpe.onnx）通常以 log-mel 作为输入（[1, 128, T] 或 [1, T, 128]）。
// - 本实现复刻 infer/lib/rmvpe.py 的关键前处理：STFT(mag) -> mel(htk=True) -> log(clamp)。
// - 目前仅实现 16k 输入、hop=160（10ms）对齐 RVC realtime 链路。

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace Ort {
class Session;
}

namespace rvc_ort {

struct Error;

class RmvpeF0 {
 public:
  RmvpeF0();
  ~RmvpeF0();

  // 载入 rmvpe.onnx（Session 由外部创建并传入；该类只读取 IO 信息并做前处理/后处理）。
  bool Init(Ort::Session* sess, Error* err);

  // 计算 f0（Hz），输出长度约为 audio_len/160 + 1。
  // - audio16k: 16k 单声道
  // - thred: rmvpe decode 阈值（对齐 python：0.03）
  bool ComputeF0Hz(const float* audio16k,
                   int32_t audio_len,
                   float thred,
                   std::vector<float>* out_f0_hz,
                   Error* err) const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rvc_ort

