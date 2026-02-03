// demo_realtime/main.cpp
// Windows 实时变声示例：miniaudio（WASAPI）采集默认输入 -> 推理线程调用 rvc_sdk_ort_process_block -> 播放默认输出。
//
// 设计要点：
// - 音频回调里只做 ring buffer 读写（避免把推理放进回调导致爆音）。
// - 推理在独立线程里按固定 block_size 执行。
// - 运行时不需要 Python；需要 3 个文件：content_encoder.onnx + synthesizer.onnx + added_*.index。

#include <atomic>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "rvc_sdk_ort.h"

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#endif

#define MINIAUDIO_IMPLEMENTATION
#include "../third_party/miniaudio/miniaudio.h"

struct Args {
  std::string enc;
  std::string syn;
  std::string index;
  std::string rmvpe;  // 可选：rmvpe.onnx（更稳定的 F0）

  bool list_devices = false;
  bool use_cuda = false;
  int32_t cap_id = -1;  // 从 --list-devices 的 Capture devices 里选
  int32_t pb_id = -1;   // 从 --list-devices 的 Playback devices 里选

  int32_t io_sr = 48000;
  int32_t model_sr = 40000;
  float block_sec = 0.25f;
  float crossfade_sec = 0.05f;
  // 说明：当前 SDK 推理为“整窗推理+裁剪”，extra_sec 越大计算量越大。
  // 实时 demo 默认取较小值以提高“跑得动”的概率；如需更接近 WebUI 的行为可手动设为 2.5。
  float extra_sec = 0.5f;

  float index_rate = 0.3f;
  int32_t sid = 0;
  int32_t up_key = 0;
  int32_t vec_dim = 768;  // v2: 768
  int32_t threads = 4;

  int32_t seconds = 0;  // 0 表示按 Enter 退出
  int32_t prefill_blocks = 2;  // 启动时先攒 N 个 block 的输出，减少首段 underflow

  // 诊断选项：用于排查“听不到声音”到底是采集/播放问题还是模型问题
  bool passthrough = false;   // 直通：不跑模型，直接把采集到的音频播出去
  bool print_levels = false;  // 打印输入/输出音量（RMS/Peak）
  float gain = 1.0f;          // passthrough 时的增益
  float gate_rms = 0.0f;      // 输入门限：低于该 RMS 则输出静音（用于避免“静音段胡言乱语”）
  float noise_scale = 0.66666f;  // 生成噪声强度（越小越稳）
  float rms_mix_rate = 1.0f;     // 音量包络混合（1=关闭；对齐 gui_v1.py 的 rms_mix_rate）
  float rmvpe_threshold = 0.03f; // RMVPE decode 阈值（对齐 python 默认）
};

static void Usage() {
  std::printf("Usage:\n");
  std::printf("  rvc_sdk_ort_realtime --enc <vec-768-layer-12.onnx> --syn <synthesizer.onnx> --index <added_*.index> [options]\n");
  std::printf("\nOptions:\n");
  std::printf("  --list-devices           List audio devices and exit\n");
  std::printf("  --cuda                   Use CUDA EP\n");
  std::printf("  --cap-id <n>             Capture device index (from --list-devices)\n");
  std::printf("  --pb-id <n>              Playback device index (from --list-devices)\n");
  std::printf("  --sid <n>                Speaker id (default 0)\n");
  std::printf("  --index-rate <f>         Retrieval blend ratio (default 0.3)\n");
  std::printf("  --up-key <n>             Pitch shift in semitones (default 0)\n");
  std::printf("  --io-sr <n>              IO sample rate (default 48000)\n");
  std::printf("  --model-sr <n>           Model sample rate (default 40000)\n");
  std::printf("  --vec-dim <256|768>      Content feature dim (default 768)\n");
  std::printf("  --block-sec <f>          Block time in seconds (default 0.25)\n");
  std::printf("  --crossfade-sec <f>      Crossfade seconds (default 0.05)\n");
  std::printf("  --extra-sec <f>          Extra seconds for padding/history (default 0.5)\n");
  std::printf("  --threads <n>            ORT intra-op threads (default 4)\n");
  std::printf("  --seconds <n>            Auto stop after N seconds (default 0 = wait for Enter)\n");
  std::printf("  --prefill-blocks <n>     Prefill N blocks before starting playback (default 2)\n");
  std::printf("  --passthrough            Bypass model, just play captured audio (debug)\n");
  std::printf("  --print-levels           Print input/output RMS+Peak (debug)\n");
  std::printf("  --gain <f>               Passthrough gain (default 1.0)\n");
  std::printf("  --gate-rms <f>           If input RMS < f, output silence (default 0 = off)\n");
  std::printf("  --noise-scale <f>        Synth noise scale (default 0.66666)\n");
  std::printf("  --rms-mix-rate <f>       RMS envelope mix rate (0~1, default 1 = off)\n");
  std::printf("  --rmvpe <path>           Use RMVPE F0 (load rmvpe.onnx)\n");
  std::printf("  --rmvpe-threshold <f>    RMVPE decode threshold (default 0.03)\n");
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
    } else if (std::strcmp(arg, "--list-devices") == 0) {
      a->list_devices = true;
    } else if (std::strcmp(arg, "--cuda") == 0) {
      a->use_cuda = true;
    } else if (std::strcmp(arg, "--cap-id") == 0 && i + 1 < argc) {
      if (!ParseInt(argv[++i], &a->cap_id)) return false;
    } else if (std::strcmp(arg, "--pb-id") == 0 && i + 1 < argc) {
      if (!ParseInt(argv[++i], &a->pb_id)) return false;
    } else if (std::strcmp(arg, "--enc") == 0 && i + 1 < argc) {
      a->enc = argv[++i];
    } else if (std::strcmp(arg, "--syn") == 0 && i + 1 < argc) {
      a->syn = argv[++i];
    } else if (std::strcmp(arg, "--index") == 0 && i + 1 < argc) {
      a->index = argv[++i];
    } else if (std::strcmp(arg, "--rmvpe") == 0 && i + 1 < argc) {
      a->rmvpe = argv[++i];
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
    } else if (std::strcmp(arg, "--seconds") == 0 && i + 1 < argc) {
      if (!ParseInt(argv[++i], &a->seconds)) return false;
    } else if (std::strcmp(arg, "--prefill-blocks") == 0 && i + 1 < argc) {
      if (!ParseInt(argv[++i], &a->prefill_blocks)) return false;
    } else if (std::strcmp(arg, "--passthrough") == 0) {
      a->passthrough = true;
    } else if (std::strcmp(arg, "--print-levels") == 0) {
      a->print_levels = true;
    } else if (std::strcmp(arg, "--gain") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->gain)) return false;
    } else if (std::strcmp(arg, "--gate-rms") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->gate_rms)) return false;
    } else if (std::strcmp(arg, "--noise-scale") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->noise_scale)) return false;
    } else if (std::strcmp(arg, "--rms-mix-rate") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->rms_mix_rate)) return false;
    } else if (std::strcmp(arg, "--rmvpe-threshold") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->rmvpe_threshold)) return false;
    } else {
      std::printf("Unknown arg: %s\n", arg);
      return false;
    }
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

static void ListDevices(ma_context* ctx) {
  ma_device_info* playback = nullptr;
  ma_uint32 playbackCount = 0;
  ma_device_info* capture = nullptr;
  ma_uint32 captureCount = 0;
  if (ma_context_get_devices(ctx, &playback, &playbackCount, &capture, &captureCount) != MA_SUCCESS) {
    std::printf("Failed to enumerate devices.\n");
    return;
  }

  std::printf("Playback devices:\n");
  for (ma_uint32 i = 0; i < playbackCount; ++i) {
    std::printf("  [%u] %s%s\n", i, playback[i].name, playback[i].isDefault ? " (default)" : "");
  }
  std::printf("Capture devices:\n");
  for (ma_uint32 i = 0; i < captureCount; ++i) {
    std::printf("  [%u] %s%s\n", i, capture[i].name, capture[i].isDefault ? " (default)" : "");
  }
}

static void PcmRbWrite(ma_pcm_rb* rb, const float* src, ma_uint32 frames, std::atomic<uint64_t>* dropped) {
  while (frames > 0) {
    ma_uint32 n = frames;
    void* p = nullptr;
    if (ma_pcm_rb_acquire_write(rb, &n, &p) != MA_SUCCESS || n == 0 || p == nullptr) {
      if (dropped) dropped->fetch_add(frames, std::memory_order_relaxed);
      return;
    }
    std::memcpy(p, src, sizeof(float) * n);
    ma_pcm_rb_commit_write(rb, n);
    src += n;
    frames -= n;
  }
}

static ma_uint32 PcmRbRead(ma_pcm_rb* rb, float* dst, ma_uint32 frames) {
  ma_uint32 got = 0;
  while (frames > 0) {
    ma_uint32 n = frames;
    void* p = nullptr;
    if (ma_pcm_rb_acquire_read(rb, &n, &p) != MA_SUCCESS || n == 0 || p == nullptr) {
      break;
    }
    std::memcpy(dst, p, sizeof(float) * n);
    ma_pcm_rb_commit_read(rb, n);
    dst += n;
    frames -= n;
    got += n;
  }
  return got;
}

struct RealtimeState {
  rvc_sdk_ort_handle_t h = nullptr;
  int32_t block_size = 0;
  int32_t io_sr = 0;
  bool passthrough = false;
  bool print_levels = false;
  float gain = 1.0f;
  float gate_rms = 0.0f;
  std::atomic<bool> in_silence{false};

  ma_pcm_rb in_rb{};
  ma_pcm_rb out_rb{};

  std::vector<float> block_in;
  std::vector<float> block_out;

  std::atomic<bool> running{false};
  std::thread worker;

  std::atomic<uint64_t> capture_callbacks{0};
  std::atomic<uint64_t> playback_callbacks{0};
  std::atomic<uint64_t> blocks{0};
  std::atomic<uint64_t> underflow_frames{0};
  std::atomic<uint64_t> dropped_in_frames{0};
  std::atomic<int32_t> last_err_code{0};
};

static void CaptureCallback(ma_device* dev, void* out, const void* in, ma_uint32 frameCount) {
  (void)out;
  auto* st = reinterpret_cast<RealtimeState*>(dev->pUserData);
  st->capture_callbacks.fetch_add(1, std::memory_order_relaxed);
  const float* in_f = reinterpret_cast<const float*>(in);
  if (in_f == nullptr || frameCount == 0) return;
  PcmRbWrite(&st->in_rb, in_f, frameCount, &st->dropped_in_frames);
}

static void PlaybackCallback(ma_device* dev, void* out, const void* in, ma_uint32 frameCount) {
  (void)in;
  auto* st = reinterpret_cast<RealtimeState*>(dev->pUserData);
  st->playback_callbacks.fetch_add(1, std::memory_order_relaxed);
  float* out_f = reinterpret_cast<float*>(out);
  if (out_f == nullptr || frameCount == 0) return;
  const ma_uint32 got = PcmRbRead(&st->out_rb, out_f, frameCount);
  if (got < frameCount) {
    std::memset(out_f + got, 0, sizeof(float) * (frameCount - got));
    st->underflow_frames.fetch_add(frameCount - got, std::memory_order_relaxed);
  }
}

static void WorkerLoop(RealtimeState* st) {
  st->running.store(true, std::memory_order_relaxed);
  const ma_uint32 bs = static_cast<ma_uint32>(st->block_size);

  // 简单性能统计：每 N 个 block 打印一次平均耗时（不做复杂 profile，避免影响实时性）
  constexpr uint64_t kStatBlocks = 20;
  uint64_t stat_blocks = 0;
  uint64_t stat_ns = 0;
  while (st->running.load(std::memory_order_relaxed)) {
    const ma_uint32 inAvail = ma_pcm_rb_available_read(&st->in_rb);
    const ma_uint32 outAvailW = ma_pcm_rb_available_write(&st->out_rb);
    if (inAvail >= bs && outAvailW >= bs) {
      (void)PcmRbRead(&st->in_rb, st->block_in.data(), bs);
      rvc_sdk_ort_error_t err{};

      const auto t0 = std::chrono::steady_clock::now();
      int32_t rc = 0;
      // gate：静音段直接输出 0，且在“静音 -> 开始说话”边界处重置 SDK 状态，避免残留伪影。
      if (st->gate_rms > 0.0f) {
        float in_rms = 0.0f, in_peak = 0.0f;
        ComputeRmsPeak(st->block_in.data(), (int32_t)bs, &in_rms, &in_peak);
        const bool silent = (in_rms < st->gate_rms);
        const bool was_silent = st->in_silence.load(std::memory_order_relaxed);
        st->in_silence.store(silent, std::memory_order_relaxed);
        if (silent) {
          std::memset(st->block_out.data(), 0, sizeof(float) * bs);
          PcmRbWrite(&st->out_rb, st->block_out.data(), bs, nullptr);
          st->blocks.fetch_add(1, std::memory_order_relaxed);
          const auto t1 = std::chrono::steady_clock::now();
          const uint64_t dt_ns = static_cast<uint64_t>(
              std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
          stat_ns += dt_ns;
          stat_blocks += 1;
          continue;
        }
        if (was_silent && !silent) {
          // 从静音恢复时，清空内部历史（避免上一段残留/叠加导致的“怪声”）
          if (st->h) (void)rvc_sdk_ort_reset_state(st->h);
        }
      }
      if (st->passthrough) {
        // 直通：用于验证采集/播放链路本身是否正常
        const float g = st->gain;
        for (ma_uint32 i = 0; i < bs; ++i) {
          float v = st->block_in[i] * g;
          if (v > 1.0f) v = 1.0f;
          if (v < -1.0f) v = -1.0f;
          st->block_out[i] = v;
        }
      } else {
        rc = rvc_sdk_ort_process_block(st->h,
                                       st->block_in.data(),
                                       st->block_size,
                                       st->block_out.data(),
                                       st->block_size,
                                       &err);
      }
      const auto t1 = std::chrono::steady_clock::now();
      const uint64_t dt_ns = static_cast<uint64_t>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
      stat_ns += dt_ns;
      stat_blocks += 1;
      if (rc != 0) {
        st->last_err_code.store(err.code, std::memory_order_relaxed);
        std::memset(st->block_out.data(), 0, sizeof(float) * bs);
      }
      PcmRbWrite(&st->out_rb, st->block_out.data(), bs, nullptr);
      st->blocks.fetch_add(1, std::memory_order_relaxed);

      if (stat_blocks >= kStatBlocks) {
        const double avg_ms = (double)stat_ns / (double)stat_blocks / 1e6;
        const double sr = (st->io_sr > 0) ? (double)st->io_sr : 1.0;
        const double block_ms = (double)st->block_size * 1000.0 / sr;
        const double denom = (avg_ms > 1e-9) ? avg_ms : 1e-9;
        const double rt = block_ms / denom;
        if (st->print_levels) {
          float in_rms = 0.0f, in_peak = 0.0f;
          float out_rms = 0.0f, out_peak = 0.0f;
          ComputeRmsPeak(st->block_in.data(), (int32_t)bs, &in_rms, &in_peak);
          ComputeRmsPeak(st->block_out.data(), (int32_t)bs, &out_rms, &out_peak);
          std::printf("[lvl] in_rms=%.4f in_peak=%.4f out_rms=%.4f out_peak=%.4f\n",
                      in_rms,
                      in_peak,
                      out_rms,
                      out_peak);
        }
        std::printf("[perf] avg process_block=%.2f ms (block=%.2f ms, rt=%.2fx) blocks=%llu\n",
                    avg_ms,
                    block_ms,
                    rt,
                    (unsigned long long)st->blocks.load(std::memory_order_relaxed));
        stat_blocks = 0;
        stat_ns = 0;
      }
      continue;
    }
    ma_sleep(1);
  }
}

int main(int argc, char** argv) {
  Args a;
  if (!ParseArgs(argc, argv, &a)) {
    Usage();
    return 2;
  }

  ma_context ctx;
  if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) {
    std::printf("Failed to init miniaudio context.\n");
    return 1;
  }

  if (a.list_devices) {
    ListDevices(&ctx);
    ma_context_uninit(&ctx);
    return 0;
  }

  // passthrough 仅用于验证 I/O 与电平，不需要任何模型/索引。
  if (!a.passthrough) {
    if (a.enc.empty() || a.syn.empty() || a.index.empty()) {
      std::printf("Missing required args.\n\n");
      Usage();
      ma_context_uninit(&ctx);
      return 2;
    }
  }

  // 选择设备（默认设备或用户指定的 index）
  ma_device_info* playbackInfos = nullptr;
  ma_uint32 playbackCount = 0;
  ma_device_info* captureInfos = nullptr;
  ma_uint32 captureCount = 0;
  ma_device_id playbackID{};
  ma_device_id captureID{};
  bool hasPlaybackID = false;
  bool hasCaptureID = false;
  if (ma_context_get_devices(&ctx, &playbackInfos, &playbackCount, &captureInfos, &captureCount) == MA_SUCCESS) {
    if (a.pb_id >= 0 && static_cast<ma_uint32>(a.pb_id) < playbackCount) {
      playbackID = playbackInfos[static_cast<ma_uint32>(a.pb_id)].id;
      hasPlaybackID = true;
    } else {
      for (ma_uint32 i = 0; i < playbackCount; ++i) {
        if (playbackInfos[i].isDefault) {
          playbackID = playbackInfos[i].id;
          hasPlaybackID = true;
          break;
        }
      }
    }

    if (a.cap_id >= 0 && static_cast<ma_uint32>(a.cap_id) < captureCount) {
      captureID = captureInfos[static_cast<ma_uint32>(a.cap_id)].id;
      hasCaptureID = true;
    } else {
      for (ma_uint32 i = 0; i < captureCount; ++i) {
        if (captureInfos[i].isDefault) {
          captureID = captureInfos[i].id;
          hasCaptureID = true;
          break;
        }
      }
    }
  }

  // 1) 初始化 SDK（passthrough 模式跳过）
  rvc_sdk_ort_error_t err{};
  rvc_sdk_ort_handle_t h = nullptr;
  int32_t bs = 0;
  if (!a.passthrough) {
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
    cfg.ep = a.use_cuda ? RVC_SDK_ORT_EP_CUDA : RVC_SDK_ORT_EP_CPU;
    cfg.intra_op_num_threads = a.threads;
    cfg.f0_min_hz = 50.0f;
    cfg.f0_max_hz = 1100.0f;
    cfg.noise_scale = a.noise_scale;
    cfg.rms_mix_rate = a.rms_mix_rate;
    cfg.f0_method = a.rmvpe.empty() ? RVC_SDK_ORT_F0_YIN : RVC_SDK_ORT_F0_RMVPE;
    cfg.rmvpe_threshold = a.rmvpe_threshold;

    h = rvc_sdk_ort_create(&cfg, &err);
    if (!h) {
      std::printf("create failed: (%d) %s\n", (int)err.code, err.message);
      ma_context_uninit(&ctx);
      return 1;
    }
    if (rvc_sdk_ort_load(h, a.enc.c_str(), a.syn.c_str(), a.index.c_str(), &err) != 0) {
      std::printf("load failed: (%d) %s\n", (int)err.code, err.message);
      rvc_sdk_ort_destroy(h);
      ma_context_uninit(&ctx);
      return 1;
    }

    if (!a.rmvpe.empty()) {
      if (rvc_sdk_ort_load_rmvpe(h, a.rmvpe.c_str(), &err) != 0) {
        std::printf("load rmvpe failed: (%d) %s\n", (int)err.code, err.message);
        rvc_sdk_ort_destroy(h);
        ma_context_uninit(&ctx);
        return 1;
      }
    }

    bs = rvc_sdk_ort_get_block_size(h);
    std::printf("Realtime started. block_size=%d @ io_sr=%d\n", (int)bs, (int)a.io_sr);
    std::printf("EP: %s\n", a.use_cuda ? "CUDA" : "CPU");
    rvc_sdk_ort_runtime_info_t info{};
    if (rvc_sdk_ort_get_runtime_info(h, &info) == 0) {
      std::printf("Synth mode: %s (total_frames=%d return_frames=%d skip_head=%d)\n",
                  info.synth_stream ? "stream" : "full",
                  (int)info.total_frames,
                  (int)info.return_frames,
                  (int)info.skip_head_frames);
    }
  } else {
    // passthrough：仅根据 io_sr/block_sec 计算 block_size（按 10ms 对齐）
    const int32_t zc = (a.io_sr > 0) ? (a.io_sr / 100) : 0;
    const int32_t block_frames = (int32_t)std::lround(a.block_sec * 100.0f);
    bs = (zc > 0) ? std::max<int32_t>(zc, block_frames * zc) : 0;
    std::printf("Realtime started. block_size=%d @ io_sr=%d\n", (int)bs, (int)a.io_sr);
    std::printf("Mode: passthrough\n");
  }

  // 2) 初始化 ring buffers（用 miniaudio 的 lock-free ring buffer）
  RealtimeState st;
  st.h = h;
  st.block_size = bs;
  st.io_sr = a.io_sr;
  st.passthrough = a.passthrough;
  st.print_levels = a.print_levels;
  st.gain = a.gain;
  st.gate_rms = a.gate_rms;
  st.block_in.assign(static_cast<size_t>(bs), 0.0f);
  st.block_out.assign(static_cast<size_t>(bs), 0.0f);

  const ma_uint32 rbFrames = static_cast<ma_uint32>(bs) * 32;
  if (ma_pcm_rb_init(ma_format_f32, 1, rbFrames, nullptr, nullptr, &st.in_rb) != MA_SUCCESS ||
      ma_pcm_rb_init(ma_format_f32, 1, rbFrames, nullptr, nullptr, &st.out_rb) != MA_SUCCESS) {
    std::printf("Failed to init ring buffers.\n");
    if (h) rvc_sdk_ort_destroy(h);
    ma_context_uninit(&ctx);
    return 1;
  }

  // 3) 打开 capture/playback 两个设备（比 duplex 更稳：推理不在回调里）
  ma_device captureDev;
  ma_device_config capCfg = ma_device_config_init(ma_device_type_capture);
  capCfg.sampleRate = static_cast<ma_uint32>(a.io_sr);
  capCfg.capture.format = ma_format_f32;
  capCfg.capture.channels = 1;
  if (hasCaptureID) capCfg.capture.pDeviceID = &captureID;
  capCfg.dataCallback = CaptureCallback;
  capCfg.pUserData = &st;
  if (ma_device_init(&ctx, &capCfg, &captureDev) != MA_SUCCESS) {
    std::printf("Failed to open capture device.\n");
    ma_pcm_rb_uninit(&st.in_rb);
    ma_pcm_rb_uninit(&st.out_rb);
    if (h) rvc_sdk_ort_destroy(h);
    ma_context_uninit(&ctx);
    return 1;
  }

  ma_device playbackDev;
  ma_device_config pbCfg = ma_device_config_init(ma_device_type_playback);
  pbCfg.sampleRate = static_cast<ma_uint32>(a.io_sr);
  pbCfg.playback.format = ma_format_f32;
  pbCfg.playback.channels = 1;
  if (hasPlaybackID) pbCfg.playback.pDeviceID = &playbackID;
  pbCfg.dataCallback = PlaybackCallback;
  pbCfg.pUserData = &st;
  if (ma_device_init(&ctx, &pbCfg, &playbackDev) != MA_SUCCESS) {
    std::printf("Failed to open playback device.\n");
    ma_device_uninit(&captureDev);
    ma_pcm_rb_uninit(&st.in_rb);
    ma_pcm_rb_uninit(&st.out_rb);
    if (h) rvc_sdk_ort_destroy(h);
    ma_context_uninit(&ctx);
    return 1;
  }

  // 4) 启动采集 -> 推理线程先跑起来攒输出 -> 再启动播放（减少启动瞬间 underflow）
  if (ma_device_start(&captureDev) != MA_SUCCESS) {
    std::printf("Failed to start capture device.\n");
    ma_device_uninit(&playbackDev);
    ma_device_uninit(&captureDev);
    ma_pcm_rb_uninit(&st.in_rb);
    ma_pcm_rb_uninit(&st.out_rb);
    if (h) rvc_sdk_ort_destroy(h);
    ma_context_uninit(&ctx);
    return 1;
  }

  st.worker = std::thread(WorkerLoop, &st);

  const int32_t prefillBlocks = (a.prefill_blocks > 0) ? a.prefill_blocks : 0;
  if (prefillBlocks > 0) {
    const ma_uint32 target = static_cast<ma_uint32>(bs) * static_cast<ma_uint32>(prefillBlocks);
    std::printf("Prefilling %d block(s)...\n", (int)prefillBlocks);
    const auto t0 = std::chrono::steady_clock::now();
    while (ma_pcm_rb_available_read(&st.out_rb) < target) {
      ma_sleep(5);
      const auto t1 = std::chrono::steady_clock::now();
      const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
      if (ms > 5000) break;  // 最多等 5s，避免设备异常时卡死
    }
  }

  if (ma_device_start(&playbackDev) != MA_SUCCESS) {
    std::printf("Failed to start playback device.\n");
    st.running.store(false, std::memory_order_relaxed);
    if (st.worker.joinable()) st.worker.join();
    ma_device_uninit(&playbackDev);
    ma_device_uninit(&captureDev);
    ma_pcm_rb_uninit(&st.in_rb);
    ma_pcm_rb_uninit(&st.out_rb);
    if (h) rvc_sdk_ort_destroy(h);
    ma_context_uninit(&ctx);
    return 1;
  }

  std::printf("Capture device state=%d, playback device state=%d\n",
              (int)ma_device_get_state(&captureDev),
              (int)ma_device_get_state(&playbackDev));

  if (a.seconds > 0) {
    std::printf("Running for %d seconds...\n", (int)a.seconds);
    ma_sleep(static_cast<ma_uint32>(a.seconds * 1000));
  } else {
    std::printf("Press Enter to stop...\n");
    (void)std::getchar();
  }

  st.running.store(false, std::memory_order_relaxed);
  if (st.worker.joinable()) st.worker.join();

  ma_device_uninit(&playbackDev);
  ma_device_uninit(&captureDev);
  ma_pcm_rb_uninit(&st.in_rb);
  ma_pcm_rb_uninit(&st.out_rb);
  ma_context_uninit(&ctx);

  std::printf("Stopped. cap_cb=%llu pb_cb=%llu blocks=%llu dropped_in=%llu underflow=%llu last_err=%d\n",
              (unsigned long long)st.capture_callbacks.load(std::memory_order_relaxed),
              (unsigned long long)st.playback_callbacks.load(std::memory_order_relaxed),
              (unsigned long long)st.blocks.load(std::memory_order_relaxed),
              (unsigned long long)st.dropped_in_frames.load(std::memory_order_relaxed),
              (unsigned long long)st.underflow_frames.load(std::memory_order_relaxed),
              (int)st.last_err_code.load(std::memory_order_relaxed));

  if (h) rvc_sdk_ort_destroy(h);
  return 0;
}
