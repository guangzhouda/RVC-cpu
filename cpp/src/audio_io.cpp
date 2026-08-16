// realtime audio IO (miniaudio WASAPI duplex)
#include "audio_io.h"

#include <algorithm>
#include <cstdio>
#include <vector>

#define MINIAUDIO_IMPLEMENTATION
#include "third_party/miniaudio.h"

namespace rvc {

namespace {
struct Impl {
    ma_device device;
    RVCEngine* engine = nullptr;
    int block = 7200;
    int input_device = -1;   // 捕获设备索引，-1=默认
    int output_device = -1;  // 回放设备索引，-1=默认
    ma_device_id in_id;      // 实际使用的设备 id
    ma_device_id out_id;
    bool in_id_set = false;
    bool out_id_set = false;
    std::vector<float> in_buf;
    std::vector<float> out_buf;
    size_t out_pos = 0;
    bool started = false;
    ma_context ctx;
    bool ctx_inited = false;
};
}

static void data_cb(ma_device* pDevice, void* pOutput, const void* pInput, ma_uint32 frameCount) {
    auto* self = (Impl*)pDevice->pUserData;
    if (!self || !self->engine) return;
    float* out = (float*)pOutput;
    const float* in = (const float*)pInput;
    self->in_buf.insert(self->in_buf.end(), in, in + frameCount);
    int block = self->block;
    while (self->in_buf.size() >= (size_t)block) {
        auto o = self->engine->process_block(self->in_buf.data(), block);
        self->out_buf.insert(self->out_buf.end(), o.begin(), o.end());
        self->in_buf.erase(self->in_buf.begin(), self->in_buf.begin() + block);
    }
    for (ma_uint32 i = 0; i < frameCount; ++i) {
        if (self->out_pos < self->out_buf.size()) out[i] = self->out_buf[self->out_pos++];
        else out[i] = 0.0f;
    }
    if (self->out_pos > 262144) {
        self->out_buf.erase(self->out_buf.begin(), self->out_buf.begin() + self->out_pos);
        self->out_pos = 0;
    }
}

void list_audio_devices() {
    ma_context ctx;
    if (ma_context_init(nullptr, 0, nullptr, &ctx) != MA_SUCCESS) {
        std::printf("[audio] context init failed\n");
        return;
    }
    ma_device_info* poutfos = nullptr;   // playback（输出）
    ma_device_info* pinfos = nullptr;    // capture（输入）
    ma_uint32 ccount = 0, pcount = 0;
    ma_context_get_devices(&ctx, &poutfos, &ccount, &pinfos, &pcount);
    std::printf("[输入] %u 个麦克风/捕获设备:\n", pcount);
    for (ma_uint32 i = 0; i < pcount; ++i)
        std::printf("  %u%s: %s\n", i, pinfos[i].isDefault ? " [默认]" : "", pinfos[i].name);
    std::printf("[输出] %u 个扬声器/回放设备:\n", ccount);
    for (ma_uint32 i = 0; i < ccount; ++i)
        std::printf("  %u%s: %s\n", i, poutfos[i].isDefault ? " [默认]" : "", poutfos[i].name);
    ma_context_uninit(&ctx);
}

LiveSession::LiveSession(const LiveConfig& cfg, RVCEngine* engine) {
    auto* p = new Impl;
    p->engine = engine;
    p->block = cfg.block_frames;
    p->input_device = cfg.input_device;
    p->output_device = cfg.output_device;
    impl_ = p;
}

LiveSession::~LiveSession() { stop(); delete (Impl*)impl_; impl_ = nullptr; }

bool LiveSession::start() {
    auto* self = (Impl*)impl_;
    if (!self || !self->engine) return false;
    // 解析设备 id：枚举后按索引取
    if (ma_context_init(nullptr, 0, nullptr, &self->ctx) != MA_SUCCESS) {
        std::printf("[audio] context init failed\n");
        return false;
    }
    self->ctx_inited = true;
    ma_device_info* poutfos = nullptr;   // playback
    ma_device_info* pinfos = nullptr;    // capture
    ma_uint32 ccount = 0, pcount = 0;
    ma_context_get_devices(&self->ctx, &poutfos, &ccount, &pinfos, &pcount);
    if (self->input_device >= 0 && (ma_uint32)self->input_device < pcount) {
        self->in_id = pinfos[self->input_device].id;
        self->in_id_set = true;
        std::printf("[audio] 输入设备: %s\n", pinfos[self->input_device].name);
    }
    if (self->output_device >= 0 && (ma_uint32)self->output_device < ccount) {
        self->out_id = poutfos[self->output_device].id;
        self->out_id_set = true;
        std::printf("[audio] 输出设备: %s\n", poutfos[self->output_device].name);
    }

    ma_device_config dc = ma_device_config_init(ma_device_type_duplex);
    dc.sampleRate = 48000;
    dc.periodSizeInFrames = self->block;
    dc.capture.pDeviceID = self->in_id_set ? &self->in_id : nullptr;
    dc.capture.format = ma_format_f32;
    dc.capture.channels = 1;
    dc.playback.pDeviceID = self->out_id_set ? &self->out_id : nullptr;
    dc.playback.format = ma_format_f32;
    dc.playback.channels = 1;
    dc.dataCallback = data_cb;
    dc.pUserData = self;
    if (ma_device_init(&self->ctx, &dc, &self->device) != MA_SUCCESS) {
        std::printf("[audio] device init failed\n");
        ma_context_uninit(&self->ctx);
        self->ctx_inited = false;
        return false;
    }
    if (ma_device_start(&self->device) != MA_SUCCESS) {
        std::printf("[audio] device start failed\n");
        ma_device_uninit(&self->device);
        return false;
    }
    self->started = true;
    running_ = true;
    std::printf("[audio] live session started (48k f32 mono duplex)\n");
    return true;
}

void LiveSession::stop() {
    auto* self = (Impl*)impl_;
    if (self && self->started) {
        ma_device_uninit(&self->device);
        self->started = false;
    }
    if (self && self->ctx_inited) {
        ma_context_uninit(&self->ctx);
        self->ctx_inited = false;
    }
    running_ = false;
}

}  // namespace rvc