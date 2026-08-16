// realtime audio IO (miniaudio WASAPI duplex)
#pragma once
#include <cstddef>
#include "engine.h"

namespace rvc {

struct LiveConfig {
    int input_device = -1;
    int output_device = -1;
    int sample_rate = 48000;
    int block_frames = 7200;
};

void list_audio_devices();

class LiveSession {
public:
    LiveSession(const LiveConfig& cfg, RVCEngine* engine);
    ~LiveSession();
    bool start();
    void stop();
    bool running() const { return running_; }
private:
    void* impl_ = nullptr;  // 实际为 Impl*，定义在 .cpp
    bool running_ = false;
};

}  // namespace rvc