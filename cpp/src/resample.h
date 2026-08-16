// sinc 重采样（与 torchaudio sinc_interp_hann 完全一致：48k<->16k 3:1/1:3 抽取）
#pragma once
#include <vector>
#include <cmath>
#include <cstdint>

namespace rvc {

// 构建 torchaudio 同款 sinc 核，返回 kernel 和中心偏移
// orig_freq/new_freq 已约分后的比值；这里直接实现 3:1 与 1:3
struct SincKernel {
    std::vector<float> k;   // 一组 polyphase 分支（new 分支）
    int width = 0;          // 对称半宽（输入域）
    int orig = 0, neu = 0;  // 约分后比值
};

inline double _i0(double x) {
    // 修正第一类零阶贝塞尔
    double s = 1.0, ds = 1.0, d = 0.0;
    do { d += 2.0; ds *= x * x / (d * d); s += ds; } while (ds > s * 1e-15);
    return s;
}

inline SincKernel build_sinc_kernel(int orig_freq, int new_freq, double rolloff = 0.99, int lowpass_filter_width = 6) {
    // 约分
    int a = orig_freq, b = new_freq;
    while (b) { int t = a % b; a = b; b = t; }
    int gcd = a;
    int orig = orig_freq / gcd, neu = new_freq / gcd;
    double base_freq = std::min(orig, neu) * rolloff;
    int width = (int)std::ceil(lowpass_filter_width * orig / base_freq);
    SincKernel sk;
    sk.width = width; sk.orig = orig; sk.neu = neu;
    // idx: [-width, -width+1, ..., -width+orig-1] / orig  (共 width+orig 个)
    int nidx = width + orig;
    std::vector<double> idx(nidx);
    for (int i = 0; i < nidx; ++i) idx[i] = (double)(-width + i) / orig;
    // 每个输出分支 m = 0..neu-1: t = -m/neu + idx
    sk.k.resize((size_t)neu * nidx);
    long double scale = base_freq / orig;
    for (int m = 0; m < neu; ++m) {
        for (int i = 0; i < nidx; ++i) {
            double t = -((double)m) / neu + idx[i];
            t *= base_freq;
            t = std::max(-(double)lowpass_filter_width, std::min((double)lowpass_filter_width, t));
            // hann 窗
            double w = std::cos(t * 3.14159265358979323846 / lowpass_filter_width / 2.0);
            w *= w;
            double s = t * 3.14159265358979323846;
            double sinct = (s == 0.0) ? 1.0 : std::sin(s) / s;
            sk.k[(size_t)m * nidx + i] = (float)(sinct * w * scale);
        }
    }
    return sk;
}

// 应用核：输入 vector<float>（单通道），输出长度 target=ceil(neu*L/orig)
// 与 torchaudio _apply_sinc_resample_kernel 一致（左侧 pad width，右侧 pad width+orig）
inline std::vector<float> apply_sinc(const SincKernel& sk, const std::vector<float>& x) {
    int L = (int)x.size();
    int target = (int)std::ceil((double)sk.neu * L / sk.orig);
    int width = sk.width;
    int nidx = width + sk.orig;
    std::vector<float> y((size_t)target, 0.0f);
    // pad: 左侧 width 个零，右侧 width+orig 个零；卷积 stride=orig
    // y[m + k*neu] ... 简化：neu==1 时一路 stride orig
    // torchaudio _apply_sinc_resample_kernel: conv1d(pad(x), kernel, stride=orig)
    // y[j] = sum_i kernel[i] * x_pad[orig*j + i]，x_pad[0]=x[-width] => x 索引 = orig*j + i - width
    for (int j = 0; j < target; ++j) {
        int m = (j % sk.neu);
        const float* kp = &sk.k[(size_t)m * nidx];
        double acc = 0.0;
        int in_start = j * sk.orig - width;  // 关键：×orig（48k->16k 时 orig=3）
        for (int i = 0; i < nidx; ++i) {
            int xi = in_start + i;
            if (xi >= 0 && xi < L) acc += (double)kp[i] * x[(size_t)xi];
        }
        y[(size_t)j] = (float)acc;
    }
    return y;
}

}  // namespace rvc