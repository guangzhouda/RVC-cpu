// RVC 变声插件 C API（DLL 导出，可被 OBS/游戏/通话宿主通过 LoadLibrary 或虚拟声卡调用）
#pragma once

#ifdef RVC_EXPORTS
#define RVC_API __declspec(dllexport)
#else
#define RVC_API __declspec(dllimport)
#endif

#ifdef __cplusplus
extern "C" {
#endif

// 创建引擎。路径可为 UTF-8；dml_device 为 DML 设备号（0=默认），<0 用 CPU
RVC_API void* rvc_engine_create(const char* hubert_onnx,
                                const char* rmvpe_onnx,
                                const char* dec_onnx,
                                const char* dec_meta,
                                int dml_device);

// 每次处理的采样数（48k 域，即 block_frame）
RVC_API int rvc_block_frames(void* handle);

// 处理一块：in/out 均 48k float32 mono，长度 = rvc_block_frames
RVC_API void rvc_process(void* handle, const float* in, float* out, int n);

// 运行时参数
RVC_API void rvc_set_auto_pitch(void* handle, int enable, int mode, double target_hz);
RVC_API void rvc_set_index_rate(void* handle, double rate);
RVC_API void rvc_set_rms_mix(void* handle, double rate);

// 销毁
RVC_API void rvc_destroy(void* handle);

RVC_API const char* rvc_version(void);

#ifdef __cplusplus
}
#endif