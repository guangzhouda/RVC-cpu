// demo_file/main.cpp
// 离线对比工具：读取 wav（自动转单声道 + 重采样到 io_sr），按 block 调用 rvc_sdk_ort_process_block，
// 然后写出 wav，便于复现/对比“怪声”来源（模型 vs 实时链路 vs 参数）。
//
// 说明：
// - 该工具不依赖 Python（运行时）。
// - 采样率/声道由 miniaudio 自动转换；输出默认 16-bit PCM WAV。

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "rvc_sdk_ort.h"

#define MINIAUDIO_IMPLEMENTATION
#include "../third_party/miniaudio/miniaudio.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>

static std::wstring Utf8OrAcpToWide_(const std::string& s) {
  if (s.empty()) return std::wstring();

  auto convert = [&](UINT cp) -> std::wstring {
    const int needed = MultiByteToWideChar(cp, 0, s.c_str(), -1, nullptr, 0);
    if (needed <= 0) return std::wstring();
    std::wstring w(static_cast<size_t>(needed - 1), L'\0');
    MultiByteToWideChar(cp, 0, s.c_str(), -1, w.data(), needed);
    return w;
  };

  std::wstring w = convert(CP_UTF8);
  if (!w.empty()) return w;
  return convert(CP_ACP);
}
#endif

struct Args {
  std::string enc;
  std::string syn;
  std::string index;
  std::string rmvpe;
  std::string in_wav;
  std::string out_wav;

  rvc_sdk_ort_ep_t ep = RVC_SDK_ORT_EP_CPU;

  int32_t io_sr = 48000;
  int32_t model_sr = 40000;
  int32_t vec_dim = 768;

  float block_sec = 0.25f;
  float crossfade_sec = 0.05f;
  float extra_sec = 0.5f;

  float index_rate = 0.3f;
  int32_t sid = 0;
  int32_t up_key = 0;
  int32_t threads = 4;

  // 诊断/鲁棒性选项
  float gate_rms = 0.0f;
  float noise_scale = 0.66666f;
  float rms_mix_rate = 1.0f;  // 1=关闭（与 realtime demo 对齐）
  float rmvpe_threshold = 0.03f;

  // 输出格式
  bool out_f32 = false;  // 默认写 s16 wav（更通用）
};

static void Usage() {
  std::printf("Usage:\n");
  std::printf("  rvc_sdk_ort_file --enc <encoder.onnx> --syn <synthesizer.onnx> --index <added_*.index> \\\n");
  std::printf("                   --in <input.wav> --out <output.wav> [options]\n");
  std::printf("\nOptions:\n");
  std::printf("  --cuda                   Use CUDA EP\n");
  std::printf("  --dml                    Use DirectML EP (GPU)\n");
  std::printf("  --rmvpe <path>            Use RMVPE F0 (load rmvpe.onnx)\n");
  std::printf("  --rmvpe-threshold <f>     RMVPE decode threshold (default 0.03)\n");
  std::printf("  --sid <n>                 Speaker id (default 0)\n");
  std::printf("  --index-rate <f>          Retrieval blend ratio (default 0.3)\n");
  std::printf("  --up-key <n>              Pitch shift in semitones (default 0)\n");
  std::printf("  --io-sr <n>               IO sample rate (default 48000)\n");
  std::printf("  --model-sr <n>            Model sample rate (default 40000)\n");
  std::printf("  --vec-dim <256|768>       Content feature dim (default 768)\n");
  std::printf("  --block-sec <f>           Block time in seconds (default 0.25)\n");
  std::printf("  --crossfade-sec <f>       Crossfade seconds (default 0.05)\n");
  std::printf("  --extra-sec <f>           Extra seconds for padding/history (default 0.5)\n");
  std::printf("  --threads <n>             ORT intra-op threads (default 4)\n");
  std::printf("  --gate-rms <f>            If input RMS < f, output silence (default 0 = off)\n");
  std::printf("  --noise-scale <f>         Synth noise scale (default 0.66666)\n");
  std::printf("  --rms-mix-rate <f>        RMS envelope mix rate (0~1, default 1 = off)\n");
  std::printf("  --out-f32                 Write float32 WAV (default is 16-bit PCM)\n");
}

static bool ParseInt(const char* s, int32_t* out) {
  if (!s || !*s) return false;
  char* end = nullptr;
  long v = std::strtol(s, &end, 10);
  if (!end || *end != '\0') return false;
  *out = static_cast<int32_t>(v);
  return true;
}

static bool ParseFloat(const char* s, float* out) {
  if (!s || !*s) return false;
  char* end = nullptr;
  float v = std::strtof(s, &end);
  if (!end || *end != '\0') return false;
  *out = v;
  return true;
}

static bool ParseArgs(int argc, char** argv, Args* a) {
  for (int i = 1; i < argc; ++i) {
    const char* arg = argv[i];
    if (std::strcmp(arg, "--help") == 0 || std::strcmp(arg, "-h") == 0) {
      return false;
    } else if (std::strcmp(arg, "--cuda") == 0) {
      a->ep = RVC_SDK_ORT_EP_CUDA;
    } else if (std::strcmp(arg, "--dml") == 0) {
      a->ep = RVC_SDK_ORT_EP_DML;
    } else if (std::strcmp(arg, "--enc") == 0 && i + 1 < argc) {
      a->enc = argv[++i];
    } else if (std::strcmp(arg, "--syn") == 0 && i + 1 < argc) {
      a->syn = argv[++i];
    } else if (std::strcmp(arg, "--index") == 0 && i + 1 < argc) {
      a->index = argv[++i];
    } else if (std::strcmp(arg, "--rmvpe") == 0 && i + 1 < argc) {
      a->rmvpe = argv[++i];
    } else if (std::strcmp(arg, "--in") == 0 && i + 1 < argc) {
      a->in_wav = argv[++i];
    } else if (std::strcmp(arg, "--out") == 0 && i + 1 < argc) {
      a->out_wav = argv[++i];
    } else if (std::strcmp(arg, "--sid") == 0 && i + 1 < argc) {
      if (!ParseInt(argv[++i], &a->sid)) return false;
    } else if (std::strcmp(arg, "--index-rate") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->index_rate)) return false;
    } else if (std::strcmp(arg, "--up-key") == 0 && i + 1 < argc) {
      if (!ParseInt(argv[++i], &a->up_key)) return false;
    } else if (std::strcmp(arg, "--io-sr") == 0 && i + 1 < argc) {
      if (!ParseInt(argv[++i], &a->io_sr)) return false;
    } else if (std::strcmp(arg, "--model-sr") == 0 && i + 1 < argc) {
      if (!ParseInt(argv[++i], &a->model_sr)) return false;
    } else if (std::strcmp(arg, "--vec-dim") == 0 && i + 1 < argc) {
      if (!ParseInt(argv[++i], &a->vec_dim)) return false;
    } else if (std::strcmp(arg, "--block-sec") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->block_sec)) return false;
    } else if (std::strcmp(arg, "--crossfade-sec") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->crossfade_sec)) return false;
    } else if (std::strcmp(arg, "--extra-sec") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->extra_sec)) return false;
    } else if (std::strcmp(arg, "--threads") == 0 && i + 1 < argc) {
      if (!ParseInt(argv[++i], &a->threads)) return false;
    } else if (std::strcmp(arg, "--gate-rms") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->gate_rms)) return false;
    } else if (std::strcmp(arg, "--noise-scale") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->noise_scale)) return false;
    } else if (std::strcmp(arg, "--rms-mix-rate") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->rms_mix_rate)) return false;
    } else if (std::strcmp(arg, "--rmvpe-threshold") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->rmvpe_threshold)) return false;
    } else if (std::strcmp(arg, "--out-f32") == 0) {
      a->out_f32 = true;
    } else {
      std::printf("Unknown arg: %s\n", arg);
      return false;
    }
  }

  // 必填参数校验
  if (a->enc.empty() || a->syn.empty() || a->index.empty() || a->in_wav.empty() || a->out_wav.empty()) {
    return false;
  }
  return true;
}

static void ComputeRmsPeak(const float* x, int32_t n, float* out_rms, float* out_peak) {
  if (!x || n <= 0) {
    if (out_rms) *out_rms = 0.0f;
    if (out_peak) *out_peak = 0.0f;
    return;
  }
  double s2 = 0.0;
  float peak = 0.0f;
  for (int32_t i = 0; i < n; ++i) {
    const float v = x[i];
    s2 += (double)v * (double)v;
    const float a = (v >= 0.0f) ? v : -v;
    if (a > peak) peak = a;
  }
  const float rms = (n > 0) ? (float)std::sqrt(s2 / (double)n) : 0.0f;
  if (out_rms) *out_rms = rms;
  if (out_peak) *out_peak = peak;
}

static int Run(const Args& a) {
  // ---- 1) 创建 SDK ----
  rvc_sdk_ort_config_t cfg{};
  cfg.io_sample_rate = a.io_sr;
  cfg.model_sample_rate = a.model_sr;
  cfg.block_time_sec = a.block_sec;
  cfg.crossfade_sec = a.crossfade_sec;
  cfg.extra_sec = a.extra_sec;
  cfg.index_rate = a.index_rate;
  cfg.sid = a.sid;
  cfg.f0_up_key = a.up_key;
  cfg.vec_dim = a.vec_dim;
  cfg.ep = a.ep;
  cfg.intra_op_num_threads = a.threads;
  cfg.f0_min_hz = 50.0f;
  cfg.f0_max_hz = 1100.0f;
  cfg.noise_scale = a.noise_scale;
  cfg.rms_mix_rate = a.rms_mix_rate;
  cfg.f0_method = a.rmvpe.empty() ? RVC_SDK_ORT_F0_YIN : RVC_SDK_ORT_F0_RMVPE;
  cfg.rmvpe_threshold = a.rmvpe_threshold;

  rvc_sdk_ort_error_t err{};
  rvc_sdk_ort_handle_t h = rvc_sdk_ort_create(&cfg, &err);
  if (!h) {
    std::fprintf(stderr, "create failed: (%d) %s\n", (int)err.code, err.message);
    return 1;
  }

  int32_t rc = rvc_sdk_ort_load(h, a.enc.c_str(), a.syn.c_str(), a.index.c_str(), &err);
  if (rc != 0) {
    std::fprintf(stderr, "load failed: (%d) %s\n", (int)err.code, err.message);
    rvc_sdk_ort_destroy(h);
    return 1;
  }
  if (!a.rmvpe.empty()) {
    rc = rvc_sdk_ort_load_rmvpe(h, a.rmvpe.c_str(), &err);
    if (rc != 0) {
      std::fprintf(stderr, "load rmvpe failed: (%d) %s\n", (int)err.code, err.message);
      rvc_sdk_ort_destroy(h);
      return 1;
    }
  }

  rvc_sdk_ort_runtime_info_t info{};
  if (rvc_sdk_ort_get_runtime_info(h, &info) == 0) {
    std::printf("io_sr=%d model_sr=%d block_size=%d total_frames=%d return_frames=%d skip_head=%d synth_stream=%d\n",
                (int)info.io_sample_rate,
                (int)info.model_sample_rate,
                (int)info.block_size,
                (int)info.total_frames,
                (int)info.return_frames,
                (int)info.skip_head_frames,
                (int)info.synth_stream);
  }

  const int32_t block = rvc_sdk_ort_get_block_size(h);
  if (block <= 0) {
    std::fprintf(stderr, "invalid block size.\n");
    rvc_sdk_ort_destroy(h);
    return 1;
  }

  // ---- 2) 解码输入 wav（转为 f32 mono + io_sr）----
  ma_decoder_config dec_cfg = ma_decoder_config_init(ma_format_f32, 1, (ma_uint32)a.io_sr);
  ma_decoder dec{};
  ma_result mr = MA_ERROR;
#ifdef _WIN32
  const std::wstring in_w = Utf8OrAcpToWide_(a.in_wav);
  mr = ma_decoder_init_file_w(in_w.c_str(), &dec_cfg, &dec);
#else
  mr = ma_decoder_init_file(a.in_wav.c_str(), &dec_cfg, &dec);
#endif
  if (mr != MA_SUCCESS) {
    std::fprintf(stderr, "failed to open input wav: %s (err=%d)\n", a.in_wav.c_str(), (int)mr);
    rvc_sdk_ort_destroy(h);
    return 1;
  }

  // ---- 3) 初始化输出 encoder ----
  ma_encoder_config enc_cfg{};
  if (a.out_f32) {
    enc_cfg = ma_encoder_config_init(ma_encoding_format_wav, ma_format_f32, 1, (ma_uint32)a.io_sr);
  } else {
    enc_cfg = ma_encoder_config_init(ma_encoding_format_wav, ma_format_s16, 1, (ma_uint32)a.io_sr);
  }
  ma_encoder enc{};
  mr = MA_ERROR;
#ifdef _WIN32
  const std::wstring out_w = Utf8OrAcpToWide_(a.out_wav);
  mr = ma_encoder_init_file_w(out_w.c_str(), &enc_cfg, &enc);
#else
  mr = ma_encoder_init_file(a.out_wav.c_str(), &enc_cfg, &enc);
#endif
  if (mr != MA_SUCCESS) {
    std::fprintf(stderr, "failed to open output wav: %s (err=%d)\n", a.out_wav.c_str(), (int)mr);
    ma_decoder_uninit(&dec);
    rvc_sdk_ort_destroy(h);
    return 1;
  }

  // ---- 4) block 推理并写出 ----
  std::vector<float> in_block(static_cast<size_t>(block));
  std::vector<float> out_block(static_cast<size_t>(block));
  std::vector<int16_t> out_s16;
  if (!a.out_f32) out_s16.resize(static_cast<size_t>(block));

  bool in_silence = false;
  uint64_t total_in_frames = 0;
  uint64_t total_out_frames = 0;

  while (true) {
    ma_uint64 got = 0;
    mr = ma_decoder_read_pcm_frames(&dec, in_block.data(), (ma_uint64)block, &got);
    if (mr != MA_SUCCESS && mr != MA_AT_END) {
      std::fprintf(stderr, "decoder read failed (err=%d)\n", (int)mr);
      break;
    }
    if (got == 0) break;
    total_in_frames += got;

    // 末尾不足一块时补 0（以便 process_block 满足固定 block_size）
    if ((int32_t)got < block) {
      std::memset(in_block.data() + got, 0, sizeof(float) * (size_t)(block - (int32_t)got));
    }

    // gate：对齐 realtime demo 行为
    if (a.gate_rms > 0.0f) {
      float rms = 0.0f, peak = 0.0f;
      ComputeRmsPeak(in_block.data(), block, &rms, &peak);
      const bool silent = (rms < a.gate_rms);
      const bool was_silent = in_silence;
      in_silence = silent;
      if (silent) {
        std::memset(out_block.data(), 0, sizeof(float) * (size_t)block);
      } else {
        if (was_silent && !silent) {
          (void)rvc_sdk_ort_reset_state(h);
        }
        rc = rvc_sdk_ort_process_block(h, in_block.data(), block, out_block.data(), block, &err);
        if (rc != 0) {
          std::fprintf(stderr, "process_block failed: (%d) %s\n", (int)err.code, err.message);
          std::memset(out_block.data(), 0, sizeof(float) * (size_t)block);
        }
      }
    } else {
      rc = rvc_sdk_ort_process_block(h, in_block.data(), block, out_block.data(), block, &err);
      if (rc != 0) {
        std::fprintf(stderr, "process_block failed: (%d) %s\n", (int)err.code, err.message);
        std::memset(out_block.data(), 0, sizeof(float) * (size_t)block);
      }
    }

    // 只写真实读到的帧数（避免输出比输入长）
    const ma_uint64 write_n = got;
    if (a.out_f32) {
      mr = ma_encoder_write_pcm_frames(&enc, out_block.data(), write_n, nullptr);
    } else {
      // float32 [-1,1] -> int16
      for (ma_uint64 i = 0; i < write_n; ++i) {
        float v = out_block[(size_t)i];
        if (v > 1.0f) v = 1.0f;
        if (v < -1.0f) v = -1.0f;
        out_s16[(size_t)i] = (int16_t)std::lrint(v * 32767.0f);
      }
      mr = ma_encoder_write_pcm_frames(&enc, out_s16.data(), write_n, nullptr);
    }
    if (mr != MA_SUCCESS) {
      std::fprintf(stderr, "encoder write failed (err=%d)\n", (int)mr);
      break;
    }
    total_out_frames += write_n;
  }

  ma_encoder_uninit(&enc);
  ma_decoder_uninit(&dec);
  rvc_sdk_ort_destroy(h);

  std::printf("done. in_frames=%llu out_frames=%llu io_sr=%d\n",
              (unsigned long long)total_in_frames,
              (unsigned long long)total_out_frames,
              (int)a.io_sr);
  return 0;
}

int main(int argc, char** argv) {
  Args a;
  if (!ParseArgs(argc, argv, &a)) {
    Usage();
    return 2;
  }
  return Run(a);
}
