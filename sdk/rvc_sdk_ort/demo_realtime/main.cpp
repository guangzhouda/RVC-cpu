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
#include <filesystem>
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
#include <windows.h>
#endif

#define MINIAUDIO_IMPLEMENTATION
#include "../third_party/miniaudio/miniaudio.h"

struct Args {
  // 可选：INI 配置文件（用于“点击即用”打包：不传任何参数也能启动）
  std::string config;

  std::string enc;
  std::string syn;
  std::string index;
  std::string rmvpe;  // 可选：rmvpe.onnx（更稳定的 F0）

  bool list_devices = false;
  rvc_sdk_ort_ep_t ep = RVC_SDK_ORT_EP_CPU;
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
  bool print_latency = false; // 打印 ring buffer 队列估算延时（不含声卡/系统内部缓冲）
  float gain = 1.0f;          // passthrough 时的增益
  float gate_rms = 0.0f;      // 输入门限：低于该 RMS 则输出静音（用于避免“静音段胡言乱语”）

  // 轻量“前置降噪/静音抑制”：逐 10ms 帧做 RMS 门控（更接近 WebUI realtime 的静音阈值逻辑）。
  // 说明：这不是深度降噪（无法在你说话时把“旁人说话”彻底消掉），但能显著减少：
  // - 静音段伪影（静音也在“赫赫/喘声/胡言乱语”）
  // - 键盘/风扇等低电平噪声触发的误检
  // 典型用法：先开 --print-levels 观察 raw_rms，再把 vad_rms 设到“底噪之上、说话之下”。
  float vad_rms = 0.0f;       // 0=关闭；>0 启用逐帧门控（建议 0.01~0.03 起步）
  float vad_floor = 0.0f;     // 静音段最小增益（0=全静音；例如 0.01 约 -40dB）
  float vad_hold_ms = 150.0f;    // hangover：判无声后再“延迟关门”这么久（ms）
  float vad_attack_ms = 10.0f;   // 开门时间常数（ms）
  float vad_release_ms = 80.0f;  // 关门时间常数（ms）
  float noise_scale = 0.66666f;  // 生成噪声强度（越小越稳）
  float rms_mix_rate = 1.0f;     // 音量包络混合（1=关闭；对齐 gui_v1.py 的 rms_mix_rate）
  float rmvpe_threshold = 0.03f; // RMVPE decode 阈值（对齐 python 默认）

  // 如果推理偶尔跑慢（rt≈1 附近），输入队列会越积越多，导致“延时越来越大”。
  // 该选项用于把延时上限钳住：当输入队列超过阈值时丢弃最旧的音频，并重置 SDK 状态。
  float max_queue_sec = 0.0f;  // 0=关闭
};

static std::filesystem::path GetExeDir() {
#ifdef _WIN32
  wchar_t buf[MAX_PATH];
  const DWORD n = GetModuleFileNameW(nullptr, buf, static_cast<DWORD>(sizeof(buf) / sizeof(buf[0])));
  if (n == 0 || n >= (DWORD)(sizeof(buf) / sizeof(buf[0]))) {
    return std::filesystem::current_path();
  }
  return std::filesystem::path(buf).parent_path();
#else
  return std::filesystem::current_path();
#endif
}

static std::string PathToUtf8(const std::filesystem::path& p) {
#ifdef _WIN32
  // C++17: u8string() 返回 std::string（UTF-8）
  return p.u8string();
#else
  return p.string();
#endif
}

static std::string ResolveMaybeRelative(const std::filesystem::path& base, const std::string& s) {
  if (s.empty()) return std::string();
  std::filesystem::path p(s);
  if (p.is_relative()) p = base / p;
  return PathToUtf8(p.lexically_normal());
}

#ifdef _WIN32
static bool IniGetString(const char* ini_path, const char* section, const char* key, std::string* out) {
  if (!ini_path || !section || !key || !out) return false;
  char buf[2048];
  buf[0] = '\0';
  const DWORD n = GetPrivateProfileStringA(section, key, "", buf, (DWORD)sizeof(buf), ini_path);
  if (n == 0) return false;
  *out = std::string(buf, buf + n);
  return true;
}

static bool IniGetInt(const char* ini_path, const char* section, const char* key, int32_t* out) {
  if (!ini_path || !section || !key || !out) return false;
  char buf[64];
  buf[0] = '\0';
  const DWORD n = GetPrivateProfileStringA(section, key, "", buf, (DWORD)sizeof(buf), ini_path);
  if (n == 0) return false;
  char* end = nullptr;
  long v = std::strtol(buf, &end, 10);
  if (end == buf) return false;
  *out = (int32_t)v;
  return true;
}

static bool IniGetFloat(const char* ini_path, const char* section, const char* key, float* out) {
  if (!ini_path || !section || !key || !out) return false;
  char buf[64];
  buf[0] = '\0';
  const DWORD n = GetPrivateProfileStringA(section, key, "", buf, (DWORD)sizeof(buf), ini_path);
  if (n == 0) return false;
  char* end = nullptr;
  float v = std::strtof(buf, &end);
  if (end == buf) return false;
  *out = v;
  return true;
}
#endif

static void ApplyIniConfigIfPresent(Args* a) {
  if (!a) return;

  // 仅在未提供必要模型参数时才读配置（避免影响命令行调参习惯）。
  if (!a->passthrough && (!a->enc.empty() && !a->syn.empty() && !a->index.empty())) return;

  const std::filesystem::path exe_dir = GetExeDir();
  std::filesystem::path ini_path = exe_dir / "rvc_realtime.ini";
  if (!a->config.empty()) {
    std::filesystem::path p(a->config);
    if (p.is_relative()) p = exe_dir / p;
    ini_path = p;
  }

  if (!std::filesystem::exists(ini_path)) {
    // 兜底：如果不存在 ini，但 models/ 下有固定命名文件，也允许直接双击运行。
    const std::filesystem::path models = exe_dir / "models";
    const std::filesystem::path enc = models / "content_encoder.onnx";
    const std::filesystem::path syn = models / "synthesizer.onnx";
    const std::filesystem::path idx0 = models / "retrieval.index";
    const std::filesystem::path rmvpe = models / "rmvpe.onnx";
    if (a->enc.empty() && std::filesystem::exists(enc)) a->enc = PathToUtf8(enc);
    if (a->syn.empty() && std::filesystem::exists(syn)) a->syn = PathToUtf8(syn);
    if (a->index.empty()) {
      if (std::filesystem::exists(idx0)) {
        a->index = PathToUtf8(idx0);
      } else if (std::filesystem::exists(models) && std::filesystem::is_directory(models)) {
        for (const auto& e : std::filesystem::directory_iterator(models)) {
          if (e.is_regular_file() && e.path().extension() == ".index") {
            a->index = PathToUtf8(e.path());
            break;
          }
        }
      }
    }
    if (a->rmvpe.empty() && std::filesystem::exists(rmvpe)) a->rmvpe = PathToUtf8(rmvpe);
    return;
  }

#ifdef _WIN32
  const std::string ini_u8 = PathToUtf8(ini_path);
  const std::filesystem::path ini_dir = ini_path.parent_path();

  // 模型文件
  std::string s;
  if (IniGetString(ini_u8.c_str(), "models", "enc", &s) && !s.empty()) a->enc = ResolveMaybeRelative(ini_dir, s);
  if (IniGetString(ini_u8.c_str(), "models", "syn", &s) && !s.empty()) a->syn = ResolveMaybeRelative(ini_dir, s);
  if (IniGetString(ini_u8.c_str(), "models", "index", &s) && !s.empty()) a->index = ResolveMaybeRelative(ini_dir, s);
  if (IniGetString(ini_u8.c_str(), "models", "rmvpe", &s) && !s.empty()) {
    const std::string p = ResolveMaybeRelative(ini_dir, s);
#  ifdef _WIN32
    if (std::filesystem::exists(std::filesystem::u8path(p))) {
#  else
    if (std::filesystem::exists(std::filesystem::path(p))) {
#  endif
      a->rmvpe = p;
    } else {
      a->rmvpe.clear();
    }
  }

  // 设备/采样率
  (void)IniGetInt(ini_u8.c_str(), "audio", "cap_id", &a->cap_id);
  (void)IniGetInt(ini_u8.c_str(), "audio", "pb_id", &a->pb_id);
  (void)IniGetInt(ini_u8.c_str(), "audio", "io_sr", &a->io_sr);

  // 推理参数
  (void)IniGetInt(ini_u8.c_str(), "rvc", "model_sr", &a->model_sr);
  (void)IniGetFloat(ini_u8.c_str(), "rvc", "block_sec", &a->block_sec);
  (void)IniGetFloat(ini_u8.c_str(), "rvc", "extra_sec", &a->extra_sec);
  (void)IniGetFloat(ini_u8.c_str(), "rvc", "crossfade_sec", &a->crossfade_sec);
  (void)IniGetInt(ini_u8.c_str(), "rvc", "prefill_blocks", &a->prefill_blocks);
  (void)IniGetFloat(ini_u8.c_str(), "rvc", "index_rate", &a->index_rate);
  (void)IniGetInt(ini_u8.c_str(), "rvc", "up_key", &a->up_key);
  (void)IniGetInt(ini_u8.c_str(), "rvc", "threads", &a->threads);
  (void)IniGetFloat(ini_u8.c_str(), "rvc", "noise_scale", &a->noise_scale);
  (void)IniGetFloat(ini_u8.c_str(), "rvc", "rms_mix_rate", &a->rms_mix_rate);
  (void)IniGetFloat(ini_u8.c_str(), "rvc", "rmvpe_threshold", &a->rmvpe_threshold);
  (void)IniGetFloat(ini_u8.c_str(), "rvc", "vad_rms", &a->vad_rms);
  (void)IniGetFloat(ini_u8.c_str(), "rvc", "vad_floor", &a->vad_floor);
  (void)IniGetFloat(ini_u8.c_str(), "rvc", "vad_hold_ms", &a->vad_hold_ms);
  (void)IniGetFloat(ini_u8.c_str(), "rvc", "vad_attack_ms", &a->vad_attack_ms);
  (void)IniGetFloat(ini_u8.c_str(), "rvc", "vad_release_ms", &a->vad_release_ms);
  (void)IniGetFloat(ini_u8.c_str(), "rvc", "max_queue_sec", &a->max_queue_sec);

  // EP（可选）：cpu/cuda/dml
  if (IniGetString(ini_u8.c_str(), "rvc", "ep", &s) && !s.empty()) {
    if (s == "cuda") a->ep = RVC_SDK_ORT_EP_CUDA;
    if (s == "dml") a->ep = RVC_SDK_ORT_EP_DML;
    if (s == "cpu") a->ep = RVC_SDK_ORT_EP_CPU;
  }
#endif
}

static void Usage() {
  std::printf("Usage:\n");
  std::printf("  rvc_sdk_ort_realtime --enc <vec-768-layer-12.onnx> --syn <synthesizer.onnx> --index <added_*.index> [options]\n");
  std::printf("\nOptions:\n");
  std::printf("  --config <path>          Load INI config (default: exe_dir/rvc_realtime.ini when model args are omitted)\n");
  std::printf("  --list-devices           List audio devices and exit\n");
  std::printf("  --cuda                   Use CUDA EP\n");
  std::printf("  --dml                    Use DirectML EP (GPU)\n");
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
  std::printf("  --print-latency          Print in/out ring buffer queued time (debug)\n");
  std::printf("  --gain <f>               Passthrough gain (default 1.0)\n");
  std::printf("  --gate-rms <f>           If input RMS < f, output silence (default 0 = off)\n");
  std::printf("  --vad-rms <f>            Frame VAD RMS threshold (10ms) to gate input (default 0 = off)\n");
  std::printf("  --vad-floor <f>          VAD floor gain (0~1, default 0)\n");
  std::printf("  --vad-hold-ms <f>        VAD hangover ms (default 150)\n");
  std::printf("  --vad-attack-ms <f>      VAD attack ms (default 10)\n");
  std::printf("  --vad-release-ms <f>     VAD release ms (default 80)\n");
  std::printf("  --noise-scale <f>        Synth noise scale (default 0.66666)\n");
  std::printf("  --rms-mix-rate <f>       RMS envelope mix rate (0~1, default 1 = off)\n");
  std::printf("  --rmvpe <path>           Use RMVPE F0 (load rmvpe.onnx)\n");
  std::printf("  --rmvpe-threshold <f>    RMVPE decode threshold (default 0.03)\n");
  std::printf("  --max-queue-sec <f>      Drop old capture audio if input queue exceeds f seconds (default 0 = off)\n");
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
    } else if (std::strcmp(arg, "--config") == 0 && i + 1 < argc) {
      a->config = argv[++i];
    } else if (std::strcmp(arg, "--list-devices") == 0) {
      a->list_devices = true;
    } else if (std::strcmp(arg, "--cuda") == 0) {
      a->ep = RVC_SDK_ORT_EP_CUDA;
    } else if (std::strcmp(arg, "--dml") == 0) {
      a->ep = RVC_SDK_ORT_EP_DML;
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
    } else if (std::strcmp(arg, "--print-latency") == 0) {
      a->print_latency = true;
    } else if (std::strcmp(arg, "--gain") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->gain)) return false;
    } else if (std::strcmp(arg, "--gate-rms") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->gate_rms)) return false;
    } else if (std::strcmp(arg, "--vad-rms") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->vad_rms)) return false;
    } else if (std::strcmp(arg, "--vad-floor") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->vad_floor)) return false;
    } else if (std::strcmp(arg, "--vad-hold-ms") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->vad_hold_ms)) return false;
    } else if (std::strcmp(arg, "--vad-attack-ms") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->vad_attack_ms)) return false;
    } else if (std::strcmp(arg, "--vad-release-ms") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->vad_release_ms)) return false;
    } else if (std::strcmp(arg, "--noise-scale") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->noise_scale)) return false;
    } else if (std::strcmp(arg, "--rms-mix-rate") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->rms_mix_rate)) return false;
    } else if (std::strcmp(arg, "--rmvpe-threshold") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->rmvpe_threshold)) return false;
    } else if (std::strcmp(arg, "--max-queue-sec") == 0 && i + 1 < argc) {
      if (!ParseFloat(argv[++i], &a->max_queue_sec)) return false;
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

static float ComputeRms(const float* x, int32_t n) {
  if (!x || n <= 0) return 0.0f;
  double s2 = 0.0;
  for (int32_t i = 0; i < n; ++i) {
    const float v = x[i];
    s2 += (double)v * (double)v;
  }
  return (float)std::sqrt(s2 / (double)n);
}

static float MsToCoeff(float ms, int32_t sr) {
  // 这里用一阶 IIR 平滑：y[n] = target + a * (y[n-1] - target)
  // a = exp(-1/(tau*sr))，tau=ms/1000。
  if (ms <= 0.0f || sr <= 0) return 0.0f;  // 0 表示“立即跟随”
  const double tau = (double)ms / 1000.0;
  const double denom = tau * (double)sr;
  if (denom <= 1e-9) return 0.0f;
  return (float)std::exp(-1.0 / denom);
}

struct VadGate {
  int32_t sr = 0;
  int32_t frame = 0;  // 每帧采样点数（固定按 10ms）
  float rms_thr = 0.0f;
  float floor_gain = 0.0f;
  int32_t hold_left = 0;
  int32_t hold_samples = 0;
  float gain = 1.0f;
  float attack_coeff = 0.0f;
  float release_coeff = 0.0f;

  bool Enabled() const { return (sr > 0) && (frame > 0) && (rms_thr > 0.0f); }

  void Configure(int32_t sample_rate,
                 float rms_threshold,
                 float floor,
                 float hold_ms,
                 float attack_ms,
                 float release_ms) {
    sr = sample_rate;
    frame = (sr > 0) ? std::max<int32_t>(1, sr / 100) : 0;  // 10ms
    rms_thr = rms_threshold;
    floor_gain = (floor < 0.0f) ? 0.0f : ((floor > 1.0f) ? 1.0f : floor);
    const double hs = (double)hold_ms * (double)sr / 1000.0;
    hold_samples = (hs > 0.0) ? (int32_t)std::lround(hs) : 0;
    if (hold_samples < 0) hold_samples = 0;
    hold_left = 0;
    gain = 1.0f;
    attack_coeff = MsToCoeff(attack_ms, sr);
    release_coeff = MsToCoeff(release_ms, sr);
  }

  void Reset() {
    hold_left = 0;
    gain = 1.0f;
  }

  // 逐帧门控：对无声帧做衰减（同时做 attack/release 平滑）。
  // 返回：该 block 内是否检测到“真有人声帧”（rms >= thr，不含 hangover）。
  bool ProcessBlock(float* x, int32_t n) {
    if (!Enabled() || !x || n <= 0) return true;

    bool any_true_voiced = false;
    for (int32_t off = 0; off < n; off += frame) {
      const int32_t len = std::min<int32_t>(frame, n - off);
      const float rms = ComputeRms(x + off, len);
      const bool true_voiced = (rms >= rms_thr);
      if (true_voiced) any_true_voiced = true;

      bool gate_open = true_voiced;
      if (true_voiced) {
        hold_left = hold_samples;
      } else if (hold_left > 0) {
        gate_open = true;
        hold_left -= len;
        if (hold_left < 0) hold_left = 0;
      }

      const float target = gate_open ? 1.0f : floor_gain;
      for (int32_t i = 0; i < len; ++i) {
        const float a = (target > gain) ? attack_coeff : release_coeff;
        gain = target + a * (gain - target);
        x[off + i] *= gain;
      }
    }
    return any_true_voiced;
  }
};

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

static void PcmRbDrop(ma_pcm_rb* rb, ma_uint32 frames) {
  while (frames > 0) {
    ma_uint32 n = frames;
    void* p = nullptr;
    if (ma_pcm_rb_acquire_read(rb, &n, &p) != MA_SUCCESS || n == 0 || p == nullptr) {
      break;
    }
    ma_pcm_rb_commit_read(rb, n);
    frames -= n;
  }
}

struct RealtimeState {
  rvc_sdk_ort_handle_t h = nullptr;
  int32_t block_size = 0;
  int32_t io_sr = 0;
  bool passthrough = false;
  bool print_levels = false;
  bool print_latency = false;
  float gain = 1.0f;
  float gate_rms = 0.0f;
  float max_queue_sec = 0.0f;
  std::atomic<bool> in_silence{false};
  VadGate vad;
  float last_out_sample = 0.0f;

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
  // 仅在错误码变化时打印一次，避免刷屏影响实时性。
  int32_t last_printed_err_code = 0;
  while (st->running.load(std::memory_order_relaxed)) {
    const ma_uint32 inAvail = ma_pcm_rb_available_read(&st->in_rb);
    const ma_uint32 outAvailW = ma_pcm_rb_available_write(&st->out_rb);

    // 如果输入队列积压太多，丢弃最旧的音频，把“延时增长”钳住。
    // 这是实时场景更实用的策略：宁可丢一点点，也不要延时越跑越大。
    if (st->max_queue_sec > 0.0f && st->io_sr > 0) {
      const double max_frames_f = st->max_queue_sec * (double)st->io_sr;
      const ma_uint32 max_frames = (max_frames_f > 0.0) ? (ma_uint32)std::lround(max_frames_f) : 0;
      if (max_frames > bs && inAvail > max_frames) {
        const ma_uint32 drop = inAvail - max_frames;
        PcmRbDrop(&st->in_rb, drop);
        if (st->h) (void)rvc_sdk_ort_reset_state(st->h);
        continue;
      }
    }

    if (inAvail >= bs && outAvailW >= bs) {
      (void)PcmRbRead(&st->in_rb, st->block_in.data(), bs);
      rvc_sdk_ort_error_t err{};

      const auto t0 = std::chrono::steady_clock::now();
      int32_t rc = 0;

      // 预先计算 raw 电平：用于 gate_rms 与调试打印（注意：后续 VAD 可能会修改 block_in）。
      float raw_rms = 0.0f, raw_peak = 0.0f;
      if (st->gate_rms > 0.0f || st->print_levels) {
        ComputeRmsPeak(st->block_in.data(), (int32_t)bs, &raw_rms, &raw_peak);
      }

      // gate：静音段直接输出 0，且在“静音 -> 开始说话”边界处重置 SDK 状态，避免残留伪影。
      bool silent = false;
      if (st->gate_rms > 0.0f) {
        silent = (raw_rms < st->gate_rms);
      }
      // VAD gate：逐 10ms 帧衰减输入；若整个 block 都未检测到人声，则同样跳过推理并输出 0。
      if (!silent && st->vad.Enabled()) {
        const bool any_true_voiced = st->vad.ProcessBlock(st->block_in.data(), (int32_t)bs);
        if (!any_true_voiced) {
          silent = true;
          // 强制关门：避免 hangover 把下一段“静音 block”误当成有声而继续跑推理。
          st->vad.hold_left = 0;
          st->vad.gain = st->vad.floor_gain;
        }
      }
      const bool was_silent = st->in_silence.load(std::memory_order_relaxed);
      st->in_silence.store(silent, std::memory_order_relaxed);
      if (silent) {
        // 轻微淡出，避免硬切造成爆点
        const int32_t fade = (st->io_sr > 0) ? std::min<int32_t>((int32_t)bs, st->io_sr / 200) : 0;  // 5ms
        if (fade > 0) {
          const float last = st->last_out_sample;
          for (int32_t i = 0; i < fade; ++i) {
            const float t = (fade > 1) ? (float)(fade - 1 - i) / (float)(fade - 1) : 0.0f;
            st->block_out[(size_t)i] = last * t;
          }
          std::memset(st->block_out.data() + fade, 0, sizeof(float) * ((size_t)bs - (size_t)fade));
        } else {
          std::memset(st->block_out.data(), 0, sizeof(float) * bs);
        }
        st->last_out_sample = 0.0f;
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
        if (err.code != last_printed_err_code) {
          last_printed_err_code = err.code;
          std::fprintf(stderr, "[err] (%d) %s\n", (int)err.code, err.message);
        }
        std::memset(st->block_out.data(), 0, sizeof(float) * bs);
      }
      PcmRbWrite(&st->out_rb, st->block_out.data(), bs, nullptr);
      if (bs > 0) {
        st->last_out_sample = st->block_out[(size_t)bs - 1];
      }
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
          std::printf("[lvl] raw_rms=%.4f raw_peak=%.4f in_rms=%.4f in_peak=%.4f out_rms=%.4f out_peak=%.4f\n",
                      raw_rms,
                      raw_peak,
                      in_rms,
                      in_peak,
                      out_rms,
                      out_peak);
        }
        if (st->print_latency && st->io_sr > 0) {
          const ma_uint32 in_q = ma_pcm_rb_available_read(&st->in_rb);
          const ma_uint32 out_q = ma_pcm_rb_available_read(&st->out_rb);
          const double sr = (double)st->io_sr;
          const double in_ms = (double)in_q * 1000.0 / sr;
          const double out_ms = (double)out_q * 1000.0 / sr;
          std::printf("[lat] in_q=%.1f ms out_q=%.1f ms est=%.1f ms\n", in_ms, out_ms, in_ms + out_ms);
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

  // “点击即用”打包：允许不传 --enc/--syn/--index，自动从 rvc_realtime.ini 或 models/ 补全。
  ApplyIniConfigIfPresent(&a);

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
      std::printf("Missing required args.\n");
      std::printf("  - Provide --enc/--syn/--index, OR\n");
      std::printf("  - Put rvc_realtime.ini next to the exe (or models/ with fixed names).\n\n");
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
    cfg.ep = a.ep;
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
    const char* ep_name = "CPU";
    if (a.ep == RVC_SDK_ORT_EP_CUDA) ep_name = "CUDA";
    if (a.ep == RVC_SDK_ORT_EP_DML) ep_name = "DML";
    std::printf("EP: %s\n", ep_name);
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
  st.print_latency = a.print_latency;
  st.gain = a.gain;
  st.gate_rms = a.gate_rms;
  st.max_queue_sec = a.max_queue_sec;
  if (a.vad_rms > 0.0f) {
    st.vad.Configure(a.io_sr, a.vad_rms, a.vad_floor, a.vad_hold_ms, a.vad_attack_ms, a.vad_release_ms);
  }
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
