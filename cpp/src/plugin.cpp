#include "plugin.h"
#include "engine.h"

#include <string>

using namespace rvc;

extern "C" {

void* rvc_engine_create(const char* hubert_onnx, const char* rmvpe_onnx,
                        const char* dec_onnx, const char* dec_meta, int dml_device) {
    EngineConfig cfg;
    cfg.hubert_onnx = hubert_onnx ? hubert_onnx : "";
    cfg.rmvpe_onnx = rmvpe_onnx ? rmvpe_onnx : "";
    cfg.dec_onnx = dec_onnx ? dec_onnx : "";
    cfg.dec_meta = dec_meta ? dec_meta : "";
    if (dml_device < 0) cfg.use_cpu = true;
    else cfg.dml_device = dml_device;
    // 默认 E 方案（无 index，auto-pitch continuous）
    cfg.index_rate = 0.0;
    cfg.auto_pitch = true;
    cfg.auto_pitch_mode = "continuous";
    try {
        return new RVCEngine(cfg);
    } catch (...) {
        return nullptr;
    }
}

int rvc_block_frames(void* handle) {
    if (!handle) return 0;
    return ((RVCEngine*)handle)->block_frame();
}

void rvc_process(void* handle, const float* in, float* out, int n) {
    if (!handle) return;
    auto o = ((RVCEngine*)handle)->process_block(in, (size_t)n);
    for (int i = 0; i < n && i < (int)o.size(); ++i) out[i] = o[i];
}

void rvc_set_auto_pitch(void* handle, int enable, int mode, double target_hz) {
    if (!handle) return;
    auto* e = (RVCEngine*)handle;
    e->set_auto_pitch(enable != 0, mode, target_hz);
}

void rvc_set_index_rate(void* handle, double rate) {
    if (!handle) return;
    ((RVCEngine*)handle)->set_index_rate(rate);
}

void rvc_set_rms_mix(void* handle, double rate) {
    if (!handle) return;
    ((RVCEngine*)handle)->set_rms_mix(rate);
}

void rvc_destroy(void* handle) {
    if (handle) delete (RVCEngine*)handle;
}

const char* rvc_version(void) {
    return "0.2.0";
}

}  // extern \"C\"