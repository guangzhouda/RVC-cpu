// rvc_sdk_ort.cpp
// C ABI 导出层：把内部 C++ 类封装成稳定接口。

#include "rvc_sdk_ort.h"

#include <cstdio>
#include <cstring>
#include <new>
#include <string>

#include "rvc_engine.h"

namespace {

static void ClearErr(rvc_sdk_ort_error_t* err) {
  if (!err) return;
  err->code = 0;
  err->message[0] = '\0';
}

static void SetErr(rvc_sdk_ort_error_t* err, int32_t code, const char* msg) {
  if (!err) return;
  err->code = code;
  if (!msg) {
    err->message[0] = '\0';
    return;
  }
  std::snprintf(err->message, sizeof(err->message), "%s", msg);
}

static void SetErrFrom(rvc_sdk_ort_error_t* err, const rvc_ort::Error& e) {
  if (!err) return;
  err->code = e.code;
  std::snprintf(err->message, sizeof(err->message), "%s", e.message.c_str());
}

}  // namespace

extern "C" {

RVC_SDK_ORT_API rvc_sdk_ort_handle_t rvc_sdk_ort_create(const rvc_sdk_ort_config_t* cfg, rvc_sdk_ort_error_t* err) {
  ClearErr(err);
  if (!cfg) {
    SetErr(err, -1, "cfg is null");
    return nullptr;
  }
  try {
    auto* engine = new rvc_ort::RvcEngine(*cfg);
    return reinterpret_cast<rvc_sdk_ort_handle_t>(engine);
  } catch (const std::bad_alloc&) {
    SetErr(err, -2, "out of memory");
    return nullptr;
  } catch (...) {
    SetErr(err, -3, "unknown exception");
    return nullptr;
  }
}

RVC_SDK_ORT_API void rvc_sdk_ort_destroy(rvc_sdk_ort_handle_t h) {
  auto* engine = reinterpret_cast<rvc_ort::RvcEngine*>(h);
  delete engine;
}

RVC_SDK_ORT_API int32_t rvc_sdk_ort_load(
    rvc_sdk_ort_handle_t h,
    const char* content_encoder_onnx,
    const char* synthesizer_onnx,
    const char* faiss_index,
    rvc_sdk_ort_error_t* err) {
  ClearErr(err);
  auto* engine = reinterpret_cast<rvc_ort::RvcEngine*>(h);
  if (!engine) {
    SetErr(err, -10, "handle is null");
    return -10;
  }
  if (!content_encoder_onnx || !synthesizer_onnx || !faiss_index) {
    SetErr(err, -11, "model/index path is null");
    return -11;
  }

  rvc_ort::Error e;
  const bool ok = engine->Load(content_encoder_onnx, synthesizer_onnx, faiss_index, &e);
  if (!ok) {
    SetErrFrom(err, e);
    return e.code != 0 ? e.code : -12;
  }
  return 0;
}

RVC_SDK_ORT_API int32_t rvc_sdk_ort_load_rmvpe(
    rvc_sdk_ort_handle_t h,
    const char* rmvpe_onnx,
    rvc_sdk_ort_error_t* err) {
  ClearErr(err);
  auto* engine = reinterpret_cast<rvc_ort::RvcEngine*>(h);
  if (!engine) {
    SetErr(err, -30, "handle is null");
    return -30;
  }
  if (!rmvpe_onnx) {
    SetErr(err, -31, "rmvpe_onnx path is null");
    return -31;
  }

  rvc_ort::Error e;
  const bool ok = engine->LoadRmvpe(rmvpe_onnx, &e);
  if (!ok) {
    SetErrFrom(err, e);
    return e.code != 0 ? e.code : -32;
  }
  return 0;
}

RVC_SDK_ORT_API int32_t rvc_sdk_ort_get_block_size(rvc_sdk_ort_handle_t h) {
  auto* engine = reinterpret_cast<rvc_ort::RvcEngine*>(h);
  if (!engine) return 0;
  return engine->GetBlockSize();
}

RVC_SDK_ORT_API int32_t rvc_sdk_ort_get_runtime_info(rvc_sdk_ort_handle_t h, rvc_sdk_ort_runtime_info_t* out_info) {
  auto* engine = reinterpret_cast<rvc_ort::RvcEngine*>(h);
  if (!engine || !out_info) return -1;
  std::memset(out_info, 0, sizeof(*out_info));
  engine->GetRuntimeInfo(out_info);
  return 0;
}

RVC_SDK_ORT_API int32_t rvc_sdk_ort_reset_state(rvc_sdk_ort_handle_t h) {
  auto* engine = reinterpret_cast<rvc_ort::RvcEngine*>(h);
  if (!engine) return -1;
  engine->ResetState();
  return 0;
}

RVC_SDK_ORT_API int32_t rvc_sdk_ort_process_block(
    rvc_sdk_ort_handle_t h,
    const float* in_mono,
    int32_t in_frames,
    float* out_mono,
    int32_t out_frames,
    rvc_sdk_ort_error_t* err) {
  ClearErr(err);
  auto* engine = reinterpret_cast<rvc_ort::RvcEngine*>(h);
  if (!engine) {
    SetErr(err, -20, "handle is null");
    return -20;
  }
  rvc_ort::Error e;
  const bool ok = engine->ProcessBlock(in_mono, in_frames, out_mono, out_frames, &e);
  if (!ok) {
    SetErrFrom(err, e);
    return e.code != 0 ? e.code : -21;
  }
  return 0;
}

}  // extern "C"
