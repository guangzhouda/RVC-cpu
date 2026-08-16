// RVC C++ - selftest PoC + convert（完整引擎）
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <random>
#include <chrono>
#include <thread>
#include <fstream>

#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include <cpu_provider_factory.h>
#include "engine.h"
#include "audio_io.h"

using namespace rvc;

// ---- 简单 WAV IO（16bit PCM 单声道）----
struct WavData { std::vector<float> mono; int sr = 0; };

static bool read_wav(const std::string& path, WavData& out) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    char hdr[44];
    f.read(hdr, 44);
    if (std::memcmp(hdr, "RIFF", 4) || std::memcmp(hdr + 8, "WAVE", 4)) return false;
    uint16_t ch = *(uint16_t*)(hdr + 22);
    int sr = *(int*)(hdr + 24);
    uint32_t data_size = *(uint32_t*)(hdr + 40);
    out.sr = sr;
    std::vector<int16_t> pcm(data_size / 2);
    f.read((char*)pcm.data(), data_size);
    out.mono.resize(pcm.size() / ch);
    for (size_t i = 0; i < out.mono.size(); ++i)
        out.mono[i] = pcm[i * ch] / 32768.0f;
    return true;
}

static bool write_wav(const std::string& path, const std::vector<float>& x, int sr) {
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    uint32_t n = (uint32_t)x.size();
    uint32_t data_bytes = n * 2;
    auto put32 = [&](uint32_t v) { f.write((char*)&v, 4); };
    f.write("RIFF", 4); put32(36 + data_bytes); f.write("WAVE", 4);
    f.write("fmt ", 4); put32(16);
    uint16_t u16 = 1; f.write((char*)&u16, 2);
    u16 = 1; f.write((char*)&u16, 2);
    int32_t i32 = sr; f.write((char*)&i32, 4);
    i32 = sr * 2; f.write((char*)&i32, 4);
    u16 = 2; f.write((char*)&u16, 2);
    u16 = 16; f.write((char*)&u16, 2);
    f.write("data", 4); put32(data_bytes);
    std::vector<int16_t> pcm(n);
    for (size_t i = 0; i < n; ++i) {
        float v = x[i];
        if (v > 0.99f) v = 0.99f; if (v < -0.99f) v = -0.99f;
        pcm[i] = (int16_t)(v * 32767.0f);
    }
    f.write((char*)pcm.data(), data_bytes);
    return true;
}

// 任意采样率 -> 48k：用 sinc 核或线性（先用线性，转换质量后续升级）
static std::vector<float> to_48k(const std::vector<float>& x, int sr) {
    if (sr == 48000) return x;
    double r = 48000.0 / sr;
    std::vector<float> y((size_t)(x.size() * r));
    for (size_t i = 0; i < y.size(); ++i) {
        double p = i / r;
        size_t i0 = (size_t)p;
        size_t i1 = (i0 + 1 < x.size()) ? i0 + 1 : x.size() - 1;
        double fr = p - i0;
        y[i] = (float)(x[i0] * (1 - fr) + x[i1] * fr);
    }
    return y;
}

static int cmd_convert(int argc, char** argv) {
    // rvc convert in.wav out.wav [--cpu] [--onnx p] [--meta p]
    std::string in_wav = argv[2], out_wav = argc > 3 ? argv[3] : "out.wav";
    EngineConfig cfg;
    for (int i = 4; i < argc; ++i) {
        if (std::string(argv[i]) == "--cpu") cfg.use_cpu = true;
        else if (i + 1 < argc && std::string(argv[i]) == "--onnx") cfg.dec_onnx = argv[++i];
        else if (i + 1 < argc && std::string(argv[i]) == "--meta") cfg.dec_meta = argv[++i];
        else if (i + 1 < argc && std::string(argv[i]) == "--hubert") cfg.hubert_onnx = argv[++i];
        else if (i + 1 < argc && std::string(argv[i]) == "--rmvpe") cfg.rmvpe_onnx = argv[++i];
        else if (i + 1 < argc && std::string(argv[i]) == "--rms-mix") cfg.rms_mix = std::atof(argv[++i]);
        else if (i + 1 < argc && std::string(argv[i]) == "--pitch") cfg.pitch = std::atoi(argv[++i]);
        else if (i + 1 < argc && std::string(argv[i]) == "--target") cfg.auto_pitch_target = std::atof(argv[++i]);
        else if (std::string(argv[i]) == "--no-auto-pitch") cfg.auto_pitch = false;
        else if (std::string(argv[i]) == "--auto-pitch") cfg.auto_pitch = true;
    }
    std::printf("[convert] 开始\n");
    WavData wav;
    if (!read_wav(in_wav, wav)) { std::printf("[error] 无法读取 %s\n", in_wav.c_str()); return 2; }
    std::printf("[convert] %s %dHz %.1fs\n", in_wav.c_str(), wav.sr, wav.mono.size() / (double)wav.sr);
    auto x48 = to_48k(wav.mono, wav.sr);
    std::printf("[convert] x48 size=%zu\n", x48.size());

    RVCEngine engine(cfg);
    std::vector<float> all;
    const int block = engine.block_frame();
    const int sr = engine.sr();
    // 逐块（不足块补零）
    size_t n = x48.size();
    size_t nblocks = (n + block - 1) / block;
    std::vector<float> buf(block, 0.f);
    auto t0 = std::chrono::steady_clock::now();
    for (size_t b = 0; b < nblocks; ++b) {
        size_t off = b * block;
        size_t take = std::min((size_t)block, n - off);
        std::fill(buf.begin(), buf.end(), 0.f);
        std::memcpy(buf.data(), x48.data() + off, take * sizeof(float));
        auto out = engine.process_block(buf.data(), take);
        all.insert(all.end(), out.begin(), out.end());
    }
    auto t1 = std::chrono::steady_clock::now();
    double sec = std::chrono::duration<double>(t1 - t0).count();
    all.resize(n);
    write_wav(out_wav, all, sr);
    std::printf("[convert] %zu -> %s (%.1fs 音频, %.2fs 处理, %dx realtime)\n",
                n, out_wav.c_str(), n / (double)sr, sec, n / (double)sr / (sec + 1e-9));
    return 0;
}

int main(int argc, char** argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    if (argc >= 2 && std::string(argv[1]) == "convert") {
        return cmd_convert(argc, argv);
    }
    if (argc >= 2 && std::string(argv[1]) == "devices") {
        list_audio_devices();
        return 0;
    }
    if (argc >= 2 && std::string(argv[1]) == "live") {
        EngineConfig cfg;
        LiveConfig lc;
        for (int i = 2; i < argc; ++i) {
            if (std::string(argv[i]) == "--cpu") cfg.use_cpu = true;
            else if (i + 1 < argc && std::string(argv[i]) == "--onnx") cfg.dec_onnx = argv[++i];
            else if (i + 1 < argc && std::string(argv[i]) == "--meta") cfg.dec_meta = argv[++i];
            else if (i + 1 < argc && std::string(argv[i]) == "--input") lc.input_device = std::atoi(argv[++i]);
            else if (i + 1 < argc && std::string(argv[i]) == "--output") lc.output_device = std::atoi(argv[++i]);
            else if (i + 1 < argc && std::string(argv[i]) == "--rms-mix") cfg.rms_mix = std::atof(argv[++i]);
            else if (i + 1 < argc && std::string(argv[i]) == "--pitch") cfg.pitch = std::atoi(argv[++i]);
            else if (i + 1 < argc && std::string(argv[i]) == "--target") cfg.auto_pitch_target = std::atof(argv[++i]);
            else if (std::string(argv[i]) == "--no-auto-pitch") cfg.auto_pitch = false;
            else if (std::string(argv[i]) == "--auto-pitch") cfg.auto_pitch = true;
        }
        std::printf("[live] 加载引擎...\n");
        RVCEngine engine(cfg);
        lc.block_frames = engine.block_frame();
        LiveSession session(lc, &engine);
        if (!session.start()) return 3;
        std::printf("[live] 实时变声中，按 Ctrl+C 退出...\n");
        while (session.running()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
        }
        return 0;
    }
    // 默认 selftest（保留 PoC）：解码器单块推理性能
    const char* onnx_path = argc > 1 ? argv[1]
        : "assets/weights/furina/onnx/stream150.int8.onnx";
    EngineConfig cfg;
    cfg.dec_onnx = onnx_path;
    cfg.use_cpu = (argc > 2 && std::string(argv[2]) == "--cpu");
    RVCEngine engine(cfg);
    std::vector<float> in(engine.block_frame(), 0.01f);
    auto t0 = std::chrono::steady_clock::now();
    constexpr int RUNS = 20;
    for (int i = 0; i < RUNS; ++i) {
        auto out = engine.process_block(in.data(), in.size());
        if (i == 0) std::printf("[output] first = %f len=%zu\n", out[0], out.size());
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / RUNS;
    std::printf("[bench] avg %.1f ms/block (budget 150ms) -> %.2fx 实时\n", ms, 150.0 / ms);
    std::printf("OK\n");
    return 0;
}