// rvc_sdk_ort.h
// Windows / C ABI：RVC 实时变声 SDK（ORT + FAISS），运行时不依赖 Python。
//
// 注意：
// - 本 SDK 以“实时变声”为目标，接口以 block（固定帧数）为单位处理。
// - 目前实现采用“滑窗整段推理 + 结果裁剪 + SOLA 对齐”（对应 docs 中阶段 A）。
// - GPU/CPU 由 ORT Execution Provider 决定（同一套 DLL 可在 CPU-only 上运行）。

#pragma once

#include <stdint.h>

#ifdef _WIN32
#  ifdef RVC_SDK_ORT_EXPORTS
#    define RVC_SDK_ORT_API __declspec(dllexport)
#  else
#    define RVC_SDK_ORT_API __declspec(dllimport)
#  endif
#else
#  define RVC_SDK_ORT_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef void* rvc_sdk_ort_handle_t;

typedef enum rvc_sdk_ort_ep_t {
  RVC_SDK_ORT_EP_CPU = 0,
  RVC_SDK_ORT_EP_CUDA = 1
} rvc_sdk_ort_ep_t;

typedef struct rvc_sdk_ort_error_t {
  int32_t code;               // 0 表示成功，其它为失败
  char message[512];          // 简短错误信息（UTF-8）
} rvc_sdk_ort_error_t;

typedef struct rvc_sdk_ort_config_t {
  // 宿主侧输入/输出采样率（SOLA 等对齐都在这个采样率上进行）
  int32_t io_sample_rate;     // 例如 48000（必须能被 100 整除）

  // 生成模型输出采样率（模型自身采样率，通常 32000/40000/48000）
  int32_t model_sample_rate;  // 例如 40000（必须能被 100 整除）

  // 实时 block 配置（秒）
  float block_time_sec;       // 例如 0.25
  float crossfade_sec;        // 例如 0.05
  float extra_sec;            // 例如 2.5（前置上下文，用于稳定音色/减少断裂）

  // 检索强度（必须保留检索）
  float index_rate;           // 0~1（建议 0.0~0.5）

  // 说话人 ID 与变调
  int32_t sid;                // 默认 0
  int32_t f0_up_key;          // 半音，允许负数

  // 内容特征维度（v1 通常 256，v2 通常 768）
  int32_t vec_dim;            // 256 或 768

  // ORT 执行后端
  rvc_sdk_ort_ep_t ep;        // CPU 或 CUDA

  // CPU 线程（仅 CPU EP 有意义；CUDA EP 仍会用到少量 CPU 线程）
  int32_t intra_op_num_threads; // <=0 表示使用 ORT 默认值

  // F0 范围（Hz）
  float f0_min_hz;            // 默认 50
  float f0_max_hz;            // 默认 1100

  // 生成噪声强度（对齐 RVC 推理：torch.randn_like(...) * noise_scale）
  // - 值越小越“稳”，但可能更干、更少细节
  // - 建议范围：0.0 ~ 0.66666（默认 0.66666）
  float noise_scale;
} rvc_sdk_ort_config_t;

// 运行时信息（用于调试/性能分析；不影响主流程）
typedef struct rvc_sdk_ort_runtime_info_t {
  int32_t io_sample_rate;
  int32_t model_sample_rate;

  int32_t block_size;         // io_sr 下的采样点数（等价于 rvc_sdk_ort_get_block_size）
  int32_t total_frames;       // 10ms 帧数（extra + block + sola_buffer + sola_search）
  int32_t return_frames;      // 10ms 帧数（block + sola_buffer + sola_search）
  int32_t skip_head_frames;   // 10ms 帧数（通常等于 extra）

  int32_t synth_stream;       // 0=普通 synthesizer.onnx（输出整窗），1=stream 导出（输出已裁剪）
} rvc_sdk_ort_runtime_info_t;

// 创建/销毁
RVC_SDK_ORT_API rvc_sdk_ort_handle_t rvc_sdk_ort_create(const rvc_sdk_ort_config_t* cfg, rvc_sdk_ort_error_t* err);
RVC_SDK_ORT_API void rvc_sdk_ort_destroy(rvc_sdk_ort_handle_t h);

// 加载模型与索引（必须在 process_block 前调用）
RVC_SDK_ORT_API int32_t rvc_sdk_ort_load(
  rvc_sdk_ort_handle_t h,
  const char* content_encoder_onnx,   // 内容特征 ONNX（输入 16k wav，输出 [T,C] 或 [C,T]）
  const char* synthesizer_onnx,       // 生成模型 ONNX（来自本仓库导出脚本）
  const char* faiss_index,            // *.index（added_*.index）
  rvc_sdk_ort_error_t* err
);

// 返回宿主每次应喂给 SDK 的 block 大小（以 io_sample_rate 的“采样点数”为单位）
RVC_SDK_ORT_API int32_t rvc_sdk_ort_get_block_size(rvc_sdk_ort_handle_t h);

// 获取运行时信息（可选）
RVC_SDK_ORT_API int32_t rvc_sdk_ort_get_runtime_info(rvc_sdk_ort_handle_t h, rvc_sdk_ort_runtime_info_t* out_info);

// 重置内部状态（清空历史缓冲/F0 cache/SOLA/rng 等）。
// 用途：宿主在“静音 -> 开始说话”的边界处调用，可减少残留与伪影。
RVC_SDK_ORT_API int32_t rvc_sdk_ort_reset_state(rvc_sdk_ort_handle_t h);

// 实时处理：输入单声道 float32 PCM（[-1,1]），输出单声道 float32 PCM（[-1,1]）
RVC_SDK_ORT_API int32_t rvc_sdk_ort_process_block(
  rvc_sdk_ort_handle_t h,
  const float* in_mono,
  int32_t in_frames,
  float* out_mono,
  int32_t out_frames,
  rvc_sdk_ort_error_t* err
);

#ifdef __cplusplus
} // extern "C"
#endif
