// RVC 实时变声引擎实现（对齐 Python apps/live_onnx.py RVCOnnxEngine.process_block）
#include "engine.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>

namespace rvc {

namespace {

std::string trim(const std::string& s) {
    size_t a = s.find_first_not_of(" \t\r\n\"");
    size_t b = s.find_last_not_of(" \t\r\n\"");
    if (a == std::string::npos) return "";
    return s.substr(a, b - a + 1);
}

// 轻量 JSON 取值：按 key 找 "key": value（仅用于 meta.json 固定结构）
double json_num(const std::string& s, const std::string& key) {
    auto p = s.find("\"" + key + "\"");
    if (p == std::string::npos) return 0;
    auto q = s.find(':', p);
    if (q == std::string::npos) return 0;
    return std::stof(s.substr(q + 1));
}

int json_int(const std::string& s, const std::string& key) { return (int)json_num(s, key); }

void die(const std::string& msg) {
    std::fprintf(stderr, "[FATAL] %s\n", msg.c_str());
    std::exit(1);
}

}  // namespace

RVCEngine::RVCEngine(const EngineConfig& cfg) : cfg_(cfg), env_(ORT_LOGGING_LEVEL_WARNING, "rvc"),
    rng_(cfg.seed) {
    Ort::SessionOptions so;
    so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);  // ALL 下 DML 出零/NaN，试 BASIC
    if (!cfg_.use_cpu) {
        const OrtApi* api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
        const OrtDmlApi* dml = nullptr;
        OrtStatus* st = api->GetExecutionProviderApi("DML", ORT_API_VERSION, (const void**)&dml);
        if (st || !dml) die("DML EP 不可用");
        OrtStatus* st2 = dml->SessionOptionsAppendExecutionProvider_DML(so, cfg_.dml_device);
        if (st2) {
            const char* msg = api->GetErrorMessage(st2);
            api->ReleaseStatus(st2);
            die(std::string("DML EP 注册失败: ") + msg);
        }
        std::printf("[ep] DirectML device_id=%d\n", cfg_.dml_device);
        // 追加 CPU EP 作为 fallback（rmvpe GRU / hubert LSTM 等算子 DML 不支持时自动回退）
        so.AppendExecutionProvider_CPU(true);
    } else {
        so.AppendExecutionProvider_CPU(true);
        std::printf("[ep] CPU\n");
    }

    // 读取 stream 配置
    {
        std::ifstream f(cfg_.dec_meta);
        if (!f) die("meta 文件不存在: " + cfg_.dec_meta);
        std::stringstream ss; ss << f.rdbuf();
        std::string meta = ss.str();
        auto sp = meta.find("\"stream\"");
        if (sp == std::string::npos) die("meta 缺少 stream 字段");
        auto sp2 = meta.find('{', sp);
        auto sp3 = meta.find('}', sp2);
        std::string st = meta.substr(sp2, sp3 - sp2 + 1);
        cfg_.zc = json_int(st, "zc");
        cfg_.block_frame = json_int(st, "block_frame");
        cfg_.crossfade_frame = json_int(st, "crossfade_frame");
        cfg_.sola_buffer_frame = json_int(st, "sola_buffer_frame");
        cfg_.sola_search_frame = json_int(st, "sola_search_frame");
        cfg_.extra_frame = json_int(st, "extra_frame");
        cfg_.phone_length = json_int(st, "phone_length");
        cfg_.skip_head = json_int(st, "skip_head");
        cfg_.sr = json_int(meta, "target_sr");
        if (cfg_.sr <= 0) cfg_.sr = 48000;
    }

    // 加载三个模型：默认全走 DML（之前 rmvpe 误判为 DML 不支持，
    // 实为 hubert feats 越界 bug；现已修复，rmvpe DML 输出与 CPU 一致）
    auto to_w = [](const std::string& p) { return std::wstring(p.begin(), p.end()); };
    Ort::SessionOptions enc_so;
    enc_so.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    enc_so.SetIntraOpNumThreads(4);
    enc_so.AppendExecutionProvider_CPU(true);
    if (cfg_.use_cpu) {
        hubert_ = std::make_unique<Ort::Session>(env_, to_w(cfg_.hubert_onnx).c_str(), enc_so);
        rmvpe_ = std::make_unique<Ort::Session>(env_, to_w(cfg_.rmvpe_onnx).c_str(), enc_so);
        dec_ = std::make_unique<Ort::Session>(env_, to_w(cfg_.dec_onnx).c_str(), enc_so);
    } else {
        hubert_ = std::make_unique<Ort::Session>(env_, to_w(cfg_.hubert_onnx).c_str(), so);
        rmvpe_ = std::make_unique<Ort::Session>(env_, to_w(cfg_.rmvpe_onnx).c_str(), so);
        dec_ = std::make_unique<Ort::Session>(env_, to_w(cfg_.dec_onnx).c_str(), so);
    }
    std::printf("[model] hubert + rmvpe + dec 均 %s 加载完成\n", cfg_.use_cpu ? "CPU" : "DML");

    // 重采样核 48k->16k
    sk48_16_ = build_sinc_kernel(cfg_.sr, 16000);

    // 缓冲
    int wav_len = cfg_.extra_frame + cfg_.crossfade_frame + cfg_.sola_search_frame + cfg_.block_frame;
    input_wav_.assign(wav_len, 0.f);
    input_wav_res_.assign((size_t)(160 * wav_len / cfg_.zc), 0.f);
    block_frame_16k_ = 160 * cfg_.block_frame / cfg_.zc;
    sola_buffer_.assign(cfg_.sola_buffer_frame, 0.f);
    fade_in_.resize(cfg_.sola_buffer_frame);
    fade_out_.resize(cfg_.sola_buffer_frame);
    for (int i = 0; i < cfg_.sola_buffer_frame; ++i) {
        double ph = 0.5 * 3.14159265358979323846 * i / (cfg_.sola_buffer_frame - 1);
        float w = (float)(std::sin(ph) * std::sin(ph));
        fade_in_[i] = w;
        fade_out_[i] = 1.f - w;
    }
    cache_pitch_.assign(1024, 0);
    cache_pitchf_.assign(1024, 0.f);

    std::printf("[stream] sr=%d block=%d (%dms) extra=%.2fs phone=%d\n",
                cfg_.sr, cfg_.block_frame, cfg_.block_frame * 1000 / cfg_.sr,
                (double)cfg_.extra_frame / cfg_.sr, cfg_.phone_length);
}

RVCEngine::~RVCEngine() = default;

// 48k 滑动窗口 -> 16k 尾部（对齐: resampler(input_wav[-block-2*zc:])[160:]）
void RVCEngine::resample_slide(const std::vector<float>& x48) {
    int win = cfg_.block_frame + 2 * cfg_.zc;
    std::vector<float> w(x48.end() - win, x48.end());
    auto y = apply_sinc(sk48_16_, w);
    int n16 = 160 * (cfg_.block_frame / cfg_.zc + 1);
    int take = std::min(n16, (int)y.size() - 160);
    std::memcpy(&input_wav_res_[input_wav_res_.size() - n16], y.data() + 160, take * sizeof(float));
    if (take < n16) {
        std::memset(&input_wav_res_[input_wav_res_.size() - n16 + take], 0, (n16 - take) * sizeof(float));
    }
}

// pitch = mel 量化后的 int64 由调用侧转换；这里返回 float 版
std::vector<float> RVCEngine::f0_post(const std::vector<float>& f0) {
    const double f0_mel_min = 1127.0 * std::log(1.0 + 50.0 / 700.0);
    const double f0_mel_max = 1127.0 * std::log(1.0 + 1100.0 / 700.0);
    std::vector<float> pitch(f0.size());
    double total_shift = cfg_.pitch + auto_shift_;
    for (size_t i = 0; i < f0.size(); ++i) {
        double v = f0[i];
        if (total_shift != 0.0 && v > 0) v = v * std::pow(2.0, total_shift / 12.0);
        double m = 1127.0 * std::log(1.0 + v / 700.0);
        if (m > 0) m = (m - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0;
        if (m <= 1.0) m = 1.0;
        if (m > 255.0) m = 255.0;
        pitch[i] = (float)std::round(m);
    }
    return pitch;
}

void RVCEngine::update_auto_pitch(const std::vector<float>& f0) {
    std::vector<double> nz;
    for (float v : f0) if (v > 0) nz.push_back(v);
    if (nz.size() < 8) return;
    for (double v : nz) med_hist_.push_back((float)v);
    if (med_hist_.size() > 80) med_hist_.erase(med_hist_.begin(), med_hist_.end() - 80);
    if (med_hist_.empty()) return;
    std::vector<float> sorted = med_hist_;
    std::sort(sorted.begin(), sorted.end());
    double med = sorted[sorted.size() / 2];
    if (med <= 0) return;
    double shift = 12.0 * std::log2(cfg_.auto_pitch_target / med);
    shift = std::max(-cfg_.auto_pitch_max, std::min(cfg_.auto_pitch_max, shift));
    if (cfg_.auto_pitch_mode != "binary") {
        auto_shift_ = 0.7 * auto_shift_ + 0.3 * shift;
        if (auto_shift_ != auto_shift_) auto_shift_ = 0.0;
    } else {
        double s = med < cfg_.auto_pitch_threshold ? cfg_.auto_pitch_male_shift : cfg_.auto_pitch_female_shift;
        auto_shift_ = 0.7 * auto_shift_ + 0.3 * s;
        if (auto_shift_ != auto_shift_) auto_shift_ = 0.0;
    }
}

std::vector<float> RVCEngine::solaa(const std::vector<float>& infer_wav) {
    const int sb = cfg_.sola_buffer_frame;
    const int ss = cfg_.sola_search_frame;
    std::vector<float> conv(infer_wav.begin(), infer_wav.begin() + sb + ss);
    double best_c = -1e18; int best_o = 0;
    for (int off = 0; off < ss + 1; ++off) {
        double nom = 0, den_a = 0, den_b = 0;
        for (int i = 0; i < sb; ++i) {
            nom += (double)conv[off + i] * sola_buffer_[i];
            den_a += (double)conv[off + i] * conv[off + i];
            den_b += (double)sola_buffer_[i] * sola_buffer_[i];
        }
        double den = std::sqrt(den_a * den_b) + 1e-8;
        double c = nom / den;
        if (c > best_c) { best_c = c; best_o = off; }
    }
    std::vector<float> out(cfg_.block_frame);
    int total = (int)infer_wav.size() - best_o;
    for (int i = 0; i < cfg_.block_frame; ++i) {
        float v = (i < total) ? infer_wav[best_o + i] : 0.f;
        if (i < sb) v = v * fade_in_[i] + sola_buffer_[i] * fade_out_[i];
        out[i] = v;
    }
    int idx = best_o + cfg_.block_frame;
    for (int i = 0; i < sb; ++i) {
        sola_buffer_[i] = (idx + i < (int)infer_wav.size()) ? infer_wav[idx + i] : 0.f;
    }
    return out;
}

// ---- rms_mix（对齐 Python live_onnx.py 4.5 段：输出响度跟随输入包络）----
static std::vector<float> rms_env(const std::vector<float>& x, int frame_len, int hop, size_t target_len) {
    // librosa.feature.rms 等价：n = 1 + (len-frame_len)/hop
    size_t len = x.size();
    size_t n = 1 + (len >= (size_t)frame_len ? (len - frame_len) / hop : 0);
    std::vector<float> rms(n);
    for (size_t t = 0; t < n; ++t) {
        size_t start = t * hop;
        size_t end = std::min(start + (size_t)frame_len, len);
        double acc = 0; size_t cnt = end - start;
        for (size_t i = start; i < end; ++i) acc += (double)x[i] * x[i];
        rms[t] = (float)std::sqrt(acc / (cnt ? cnt : 1));
    }
    // F.interpolate(线性) 到 target_len+1，取 [:-1]
    std::vector<float> out(target_len);
    for (size_t i = 0; i < target_len; ++i) {
        // 映射：源帧索引 = i * (n-1)/(target_len) （align_corners 不精确，但线性够用）
        double pos = (n <= 1) ? 0.0 : (double)i * (n - 1) / (double)target_len;
        size_t i0 = (size_t)pos;
        size_t i1 = std::min(i0 + 1, n - 1);
        double fr = pos - i0;
        out[i] = rms[i0] * (float)(1.0 - fr) + rms[i1] * (float)fr;
    }
    return out;
}
// ---- process_block: 对齐 Python live_onnx.py ----
std::vector<float> RVCEngine::process_block(const float* indata, size_t n) {
    const int block = cfg_.block_frame;
    const int zc = cfg_.zc;

    // 1) 48k 滑动缓冲
    std::memmove(&input_wav_[0], &input_wav_[block], (input_wav_.size() - block) * sizeof(float));
    std::memcpy(&input_wav_[input_wav_.size() - block], indata, n * sizeof(float));

    // 1b) 16k 域滑动 + 尾部重采样
    std::memmove(&input_wav_res_[0], &input_wav_res_[block_frame_16k_],
                 (input_wav_res_.size() - block_frame_16k_) * sizeof(float));
    int n16 = 160 * (block / zc + 1);
    {
        int win = block + 2 * zc;
        std::vector<float> w(input_wav_.end() - win, input_wav_.end());
        auto y = apply_sinc(sk48_16_, w);
        int take = std::min(n16, (int)y.size() - 160);
        std::memcpy(&input_wav_res_[input_wav_res_.size() - n16], y.data() + 160, take * sizeof(float));
        if (take < n16)
            std::memset(&input_wav_res_[input_wav_res_.size() - n16 + take], 0, (n16 - take) * sizeof(float));
    }

    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

    // 2) HuBERT 特征
    size_t T = input_wav_res_.size();
    size_t pad = (640 - T % 640) % 640;
    std::vector<float> srcp(T + pad);
    std::memcpy(srcp.data(), input_wav_res_.data(), T * sizeof(float));
    std::memset(srcp.data() + T, 0, pad * sizeof(float));
    int64_t sh_src[2] = {1, (int64_t)srcp.size()};
    Ort::Value t_src = Ort::Value::CreateTensor<float>(mem, srcp.data(), srcp.size(), sh_src, 2);
    const char* h_in[] = {"source"};
    Ort::Value h_vals[] = {std::move(t_src)};
    const char* h_out_names[] = {"feats"};
    auto h_out = hubert_->Run(Ort::RunOptions{nullptr}, h_in, h_vals, 1, h_out_names, 1);
    float* hdata = h_out[0].GetTensorMutableData<float>();
    size_t nf = (T + 319) / 320;  // ceil(T/320)
    auto hsh = h_out[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t F = (size_t)hsh[2];
    size_t H = (size_t)hsh[1];  // hubert 实际输出帧数（通常 nf-1）
    size_t fc = nf < H ? nf : H;  // 取 min(nf, H)，对齐 Python feats_raw[:, :nf]
    std::vector<float> feats((fc + 1) * F);
    std::memcpy(feats.data(), hdata, fc * F * sizeof(float));
    std::memcpy(feats.data() + fc * F, hdata + (fc - 1) * F, F * sizeof(float));  // 复制末帧

    // 3) RMVPE 音高
    int b16 = block_frame_16k_;
    int fex = 5120 * ((b16 + 800 - 1) / 5120 + 1) - 160;
    std::vector<float> win(input_wav_res_.end() - fex, input_wav_res_.end());
    int64_t sh_audio[2] = {1, (int64_t)win.size()};
    Ort::Value t_audio = Ort::Value::CreateTensor<float>(mem, win.data(), win.size(), sh_audio, 2);
    const char* r_in[] = {"audio"};
    Ort::Value r_vals[] = {std::move(t_audio)};
    const char* r_out_names[] = {"hidden"};
    std::vector<Ort::Value> r_out;
    try {
        r_out = rmvpe_->Run(Ort::RunOptions{nullptr}, r_in, r_vals, 1, r_out_names, 1);
    } catch (const Ort::Exception& e) {
        std::printf("[rmvpe] EXC: %s\n", e.what());
        return std::vector<float>(cfg_.block_frame, 0.f);
    }
    float* hidden = r_out[0].GetTensorMutableData<float>();
    auto rsh = r_out[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t h_frames = (size_t)rsh[1], h_dim = (size_t)rsh[2];
    size_t nfr = fex / 160 + 1;
    std::vector<float> f0(nfr, 0.f);
    for (size_t t = 0; t < nfr && t < h_frames; ++t) {
        const float* row = hidden + t * h_dim;
        float maxx = 0.f;
        for (size_t k = 0; k < h_dim; ++k) if (row[k] > maxx) maxx = row[k];
        int cmax = 0;
        for (size_t k = 1; k < h_dim; ++k) if (row[k] > row[cmax]) cmax = (int)k;
        int center = cmax + 4, start = center - 4, end = center + 5;
        double num = 0, den = 0;
        for (int k = start; k < end; ++k) {
            float sal = (k >= 4 && k - 4 < (int)h_dim) ? row[k - 4] : 0.f;
            float cents = 20.0f * (k - 4) + 1997.3794084376191f;  // 对齐 _rmvpe_cents 的 pad 后索引
            num += sal * cents;
            den += sal;
        }
        double devided = (den > 0) ? num / den : 0.0;
        if (maxx <= 0.03f) devided = 0.0;
        double f = 10.0 * std::pow(2.0, devided / 1200.0);
        f0[t] = (std::fabs(f - 10.0) < 1e-9) ? 0.f : (float)f;
    }

    // 4) pitch/pitchf 计算（先算，用上一个 auto_shift），再更新 auto_shift（对齐 Python 顺序）
    const double f0_mel_min = 1127.0 * std::log(1.0 + 50.0 / 700.0);
    const double f0_mel_max = 1127.0 * std::log(1.0 + 1100.0 / 700.0);
    double total_shift = cfg_.pitch;
    if (cfg_.auto_pitch) total_shift += auto_shift_;
    std::vector<int64_t> pitch(f0.size());
    std::vector<float> pitchf(f0.size());
    for (size_t i = 0; i < f0.size(); ++i) {
        double v = f0[i];
        if (total_shift != 0.0 && v > 0) v = v * std::pow(2.0, total_shift / 12.0);
        pitchf[i] = (float)v;
        double m2 = 1127.0 * std::log(1.0 + v / 700.0);
        if (m2 > 0) m2 = (m2 - f0_mel_min) * 254.0 / (f0_mel_max - f0_mel_min) + 1.0;
        if (m2 <= 1.0) m2 = 1.0;
        if (m2 > 255.0) m2 = 255.0;
        pitch[i] = (int64_t)std::lround(m2);
    }
    // 4b) 更新 auto_pitch（对齐 Python：先算 pitch 用旧 shift，再更新）
    if (cfg_.auto_pitch) update_auto_pitch(f0);

    // 5) pitch cache
    int shift = b16 / 160;
    std::memmove(&cache_pitch_[0], &cache_pitch_[shift], (cache_pitch_.size() - shift) * sizeof(int64_t));
    std::memmove(&cache_pitchf_[0], &cache_pitchf_[shift], (cache_pitchf_.size() - shift) * sizeof(float));
    size_t np = pitch.size();
    if (np > 4) {
        size_t len = np - 4;
        int64_t* cpd = cache_pitch_.data() + cache_pitch_.size() - len;
        float* cpf = cache_pitchf_.data() + cache_pitchf_.size() - len;
        for (size_t i = 0; i < len; ++i) { cpd[i] = pitch[3 + i]; cpf[i] = pitchf[3 + i]; }
    }
    std::vector<int64_t> c_pitch(cache_pitch_.end() - cfg_.phone_length, cache_pitch_.end());
    std::vector<float> c_pitchf(cache_pitchf_.end() - cfg_.phone_length, cache_pitchf_.end());

    // 6) 特征 2x 上采样（对齐 F.interpolate scale_factor=2, align_corners=False）
    size_t nf1 = fc + 1;
    std::vector<float> feats_i((size_t)cfg_.phone_length * F);
    for (size_t t = 0; t < (size_t)cfg_.phone_length; ++t) {
        size_t s = (size_t)std::floor((double)t / 2.0);
        if (s >= nf1) s = nf1 - 1;
        std::memcpy(feats_i.data() + t * F, feats.data() + s * F, F * sizeof(float));
    }

    // 7) rnd + 解码器
    std::vector<float> rnd(192 * 45);
    {
        std::normal_distribution<float> nd(0.f, 1.f);
        for (auto& v : rnd) v = nd(rng_);
    }
    std::vector<int64_t> plen = {(int64_t)cfg_.phone_length};
    std::vector<int64_t> sid = {0};
    int64_t sh_p[3] = {1, cfg_.phone_length, (int64_t)F};
    int64_t sh_1[1] = {1};
    int64_t sh_p2[2] = {1, cfg_.phone_length};
    int64_t sh_r[3] = {1, 192, 45};
    Ort::Value t_phone = Ort::Value::CreateTensor<float>(mem, feats_i.data(), feats_i.size(), sh_p, 3);
    Ort::Value t_plen  = Ort::Value::CreateTensor<int64_t>(mem, plen.data(), plen.size(), sh_1, 1);
    Ort::Value t_pitch = Ort::Value::CreateTensor<int64_t>(mem, c_pitch.data(), c_pitch.size(), sh_p2, 2);
    Ort::Value t_pf    = Ort::Value::CreateTensor<float>(mem, c_pitchf.data(), c_pitchf.size(), sh_p2, 2);
    Ort::Value t_sid   = Ort::Value::CreateTensor<int64_t>(mem, sid.data(), sid.size(), sh_1, 1);
    Ort::Value t_rnd   = Ort::Value::CreateTensor<float>(mem, rnd.data(), rnd.size(), sh_r, 3);
    const char* d_in[] = {"phone", "phone_lengths", "pitch", "pitchf", "sid", "rnd"};
    Ort::Value d_vals[] = {std::move(t_phone), std::move(t_plen), std::move(t_pitch),
                           std::move(t_pf), std::move(t_sid), std::move(t_rnd)};
    const char* d_out_names[] = {"audio"};
    auto d_out = dec_->Run(Ort::RunOptions{nullptr}, d_in, d_vals, 6, d_out_names, 1);
    float* audio = d_out[0].GetTensorMutableData<float>();
    auto dsh = d_out[0].GetTensorTypeAndShapeInfo().GetShape();
    size_t infer_len = 1;
    for (auto s : dsh) infer_len *= (size_t)s;
    std::vector<float> infer_wav(audio, audio + infer_len);

    // 7b) rms_mix：输出响度跟随输入包络（对齐 Python）
    if (cfg_.rms_mix > 0.0 && cfg_.rms_mix < 1.0) {
        int frame_len = 4 * cfg_.zc;   // 1920
        int hop = cfg_.zc;             // 480
        size_t isz = infer_wav.size();
        // src = input_wav[extra_frame : extra_frame + infer_len]
        std::vector<float> src(input_wav_.begin() + cfg_.extra_frame,
                               input_wav_.begin() + cfg_.extra_frame + isz);
        auto rms1 = rms_env(src, frame_len, hop, isz);
        auto rms2 = rms_env(infer_wav, frame_len, hop, isz);
        double ratio = 1.0 - cfg_.rms_mix;
        for (size_t i = 0; i < isz; ++i) {
            double r2 = std::max((double)rms2[i], 1e-3);
            infer_wav[i] *= (float)std::pow((double)rms1[i] / r2, ratio);
        }
    }

    // 8) SOLA
    return solaa(infer_wav);
}

}  // namespace rvc