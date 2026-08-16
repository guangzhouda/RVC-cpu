// RVC 实时变声引擎（C++/ONNX/DML）
#pragma once
#include <string>
#include <vector>
#include <random>
#include <memory>
#include <cstdint>

#include <onnxruntime_cxx_api.h>
#include <dml_provider_factory.h>
#include <cpu_provider_factory.h>
#include "resample.h"

namespace rvc {

struct EngineConfig {
    std::string hubert_onnx = "assets/hubert/_exp_hubert.onnx";
    std::string rmvpe_onnx = "assets/rmvpe/rmvpe.onnx";
    std::string dec_onnx = "assets/weights/furina/onnx/stream150.onnx";
    std::string dec_meta = "assets/weights/furina/onnx/stream150.onnx.json";
    // 与 meta.json stream 字段一一对应
    int sr = 48000, zc = 480, block_frame = 7200, crossfade_frame = 2400,
        sola_buffer_frame = 1920, sola_search_frame = 480, extra_frame = 120000,
        phone_length = 271, skip_head = 250;
    bool if_f0 = 1;
    // 音高
    bool auto_pitch = true;
    std::string auto_pitch_mode = "continuous";  // binary | continuous | cdf
    double auto_pitch_target = 220.0;
    double auto_pitch_threshold = 165.0;
    double auto_pitch_male_shift = 12.0, auto_pitch_female_shift = 0.0;
    double auto_pitch_max = 14.0;
    int pitch = 0;
    // 索引/响度（本期先存参数，后续实现）
    std::string index_path;
    double index_rate = 0.0;
    double rms_mix = 0.0;
    // 运行
    int dml_device = 0;
    bool use_cpu = false;
    uint32_t seed = 1234;
};

class RVCEngine {
public:
    explicit RVCEngine(const EngineConfig& cfg);
    ~RVCEngine();

    // 输入 block_frame 采样 @48k，返回 block_frame 变声采样 @48k
    std::vector<float> process_block(const float* indata, size_t n);

    const EngineConfig& cfg() const { return cfg_; }
    int block_frame() const { return cfg_.block_frame; }
    int sr() const { return cfg_.sr; }

    // 运行时热调
    void set_auto_pitch(bool enable, int mode, double target_hz) {
        cfg_.auto_pitch = enable;
        cfg_.auto_pitch_mode = (mode == 0) ? "binary" : (mode == 2 ? "cdf" : "continuous");
        cfg_.auto_pitch_target = target_hz;
    }
    void set_index_rate(double rate) { cfg_.index_rate = rate; }
    void set_rms_mix(double rate) { cfg_.rms_mix = rate; }

private:
    // 48k->16k 重采样（滑动窗口）
    void resample_slide(const std::vector<float>& x48);
    std::vector<float> f0_post(const std::vector<float>& f0);
    void update_auto_pitch(const std::vector<float>& f0);
    std::vector<float> solaa(const std::vector<float>& infer_wav);

    EngineConfig cfg_;
    int target_sr_ = 0;  // meta 里解析出的目标采样率
    Ort::Env env_;
    std::unique_ptr<Ort::Session> hubert_, rmvpe_, dec_;
    SincKernel sk48_16_;

    // 缓冲（48k 与 16k 双域，与 Python 一致）
    std::vector<float> input_wav_;      // extra+crossfade+sola_search+block @48k
    std::vector<float> input_wav_res_;  // 对应的 16k 域
    int block_frame_16k_ = 0;
    std::vector<float> sola_buffer_;
    std::vector<float> fade_in_, fade_out_;

    // pitch 缓存
    std::vector<int64_t> cache_pitch_;
    std::vector<float> cache_pitchf_;

    // auto-pitch 状态
    double auto_shift_ = 0.0;
    std::vector<float> med_hist_;

    std::mt19937 rng_;
};

}  // namespace rvc