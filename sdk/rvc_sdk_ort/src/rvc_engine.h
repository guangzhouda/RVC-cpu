// rvc_engine.h
// 内部实现：ORT + FAISS + 实时状态机（缓冲/重采样/SOLA）+ F0（YIN）

#pragma once

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "rvc_sdk_ort.h"

namespace rvc_ort {

struct Error {
  int32_t code = 0;
  std::string message;
};

// 说明：这里把“时间单位”统一成 10ms 的 frame（1 秒 = 100 帧）。
struct FramePlan {
  int32_t io_sr = 0;
  int32_t model_sr = 0;

  int32_t zc_io = 0;      // io_sr / 100
  int32_t zc_model = 0;   // model_sr / 100

  int32_t extra_frames = 0;
  int32_t block_frames = 0;
  int32_t crossfade_frames = 0;
  int32_t sola_buffer_frames = 0; // min(crossfade_frames, 4)
  int32_t sola_search_frames = 1; // 固定 10ms

  int32_t total_frames = 0;       // extra + block + sola_buffer + sola_search
  int32_t return_frames = 0;      // block + sola_buffer + sola_search
  int32_t skip_head_frames = 0;   // extra

  int32_t block_size_io = 0;      // block_frames * zc_io（宿主每次输入输出采样点数）
  int32_t total_size_io = 0;      // total_frames * zc_io
  int32_t total_size_16k = 0;     // total_frames * 160

  int32_t sola_buffer_size_io = 0;  // sola_buffer_frames * zc_io
  int32_t sola_search_size_io = 0;  // sola_search_frames * zc_io
};

class RvcEngine {
 public:
  explicit RvcEngine(const rvc_sdk_ort_config_t& cfg);
  ~RvcEngine();

  int32_t GetBlockSize() const;
  void GetRuntimeInfo(rvc_sdk_ort_runtime_info_t* out_info) const;
  void ResetState();

 bool Load(const std::string& content_encoder_onnx,
            const std::string& synthesizer_onnx,
            const std::string& faiss_index,
            Error* err);

  bool ProcessBlock(const float* in_mono,
                    int32_t in_frames,
                    float* out_mono,
                    int32_t out_frames,
                    Error* err);

 private:
 bool InitPlan_(Error* err);
 bool InitSessions_(Error* err);
 bool DetectSynthMode_(Error* err);

  // 一次推理（整窗），返回 io_sr 下的 infer_wav（长度 return_frames * zc_io）
  bool InferWindow_(std::vector<float>* infer_wav_io, Error* err);

  // 内容特征（16k）-> feats20（20ms 帧）-> 检索融合 -> 上采样成 feats10（10ms 帧）
  bool ExtractAndRetrieve_(std::vector<float>* feats10, int32_t* p_len, Error* err);

  bool ComputeF0WithCache_(int32_t p_len,
                           std::vector<int64_t>* pitch,
                           std::vector<float>* pitchf,
                           Error* err);

  void UpdateInputBuffers_(const float* in_mono, int32_t in_frames);

 private:
  rvc_sdk_ort_config_t cfg_;
  FramePlan plan_;

  // 实时输入缓冲（io_sr）与对应 16k 缓冲（对齐到 10ms frame）
  std::vector<float> input_io_;
  std::vector<float> input_16k_;

  // SOLA 状态
  std::vector<float> sola_buffer_;
  std::vector<float> fade_in_win_;
  std::vector<float> fade_out_win_;

  // F0 cache（按 10ms frame 存，长度固定）
  std::vector<int64_t> cache_pitch_;
  std::vector<float> cache_pitchf_;

  // 随机噪声（给 synthesizer onnx 的 rnd）
  std::mt19937 rng_;
  std::normal_distribution<float> norm01_;

  // 不在头文件里直接暴露 ORT/FAISS 类型，避免污染 include。
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace rvc_ort
