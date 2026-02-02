// demo/main.cpp
// 最小 demo：加载模型/索引，跑几次空音频 block。

#include <cstdio>
#include <cstdlib>
#include <vector>

#include "rvc_sdk_ort.h"

static void Usage() {
  std::printf("Usage:\n");
  std::printf("  rvc_sdk_ort_demo <content_encoder.onnx> <synthesizer.onnx> <model.index>\n");
}

int main(int argc, char** argv) {
  if (argc < 4) {
    Usage();
    return 2;
  }

  const char* enc = argv[1];
  const char* syn = argv[2];
  const char* idx = argv[3];

  rvc_sdk_ort_config_t cfg{};
  cfg.io_sample_rate = 48000;
  cfg.model_sample_rate = 40000;
  cfg.block_time_sec = 0.25f;
  cfg.crossfade_sec = 0.05f;
  cfg.extra_sec = 2.5f;
  cfg.index_rate = 0.3f;
  cfg.sid = 0;
  cfg.f0_up_key = 0;
  cfg.vec_dim = 768;
  cfg.ep = RVC_SDK_ORT_EP_CPU; // demo 默认 CPU
  cfg.intra_op_num_threads = 4;
  cfg.f0_min_hz = 50.0f;
  cfg.f0_max_hz = 1100.0f;
  cfg.noise_scale = 0.66666f;

  rvc_sdk_ort_error_t err{};
  rvc_sdk_ort_handle_t h = rvc_sdk_ort_create(&cfg, &err);
  if (!h) {
    std::printf("create failed: (%d) %s\n", (int)err.code, err.message);
    return 1;
  }

  int32_t rc = rvc_sdk_ort_load(h, enc, syn, idx, &err);
  if (rc != 0) {
    std::printf("load failed: (%d) %s\n", (int)err.code, err.message);
    rvc_sdk_ort_destroy(h);
    return 1;
  }

  const int32_t block = rvc_sdk_ort_get_block_size(h);
  std::printf("block_size = %d samples @ %d Hz\n", (int)block, (int)cfg.io_sample_rate);

  std::vector<float> in(block, 0.0f);
  std::vector<float> out(block, 0.0f);

  for (int i = 0; i < 3; ++i) {
    rc = rvc_sdk_ort_process_block(h, in.data(), block, out.data(), block, &err);
    if (rc != 0) {
      std::printf("process failed: (%d) %s\n", (int)err.code, err.message);
      rvc_sdk_ort_destroy(h);
      return 1;
    }
    std::printf("process ok (%d)\n", i);
  }

  rvc_sdk_ort_destroy(h);
  std::printf("done\n");
  return 0;
}
