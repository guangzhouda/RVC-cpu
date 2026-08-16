# -*- coding: utf-8 -*-
"""RVCPlugin: 实时变声插件封装（基于 apps.live_onnx.RVCOnnxEngine）。"""
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from apps.live_onnx import RVCOnnxEngine  # noqa: E402

PLANS = {
    # 名称 -> (onnx, meta, index, index_rate, rms_mix)
    "E": ("assets/weights/furina/onnx/stream150.int8.onnx",
          "assets/weights/furina/onnx/stream150.onnx.json",
          None, 0.0, None),
    "F": ("assets/weights/furina/onnx/stream150.onnx",
          "assets/weights/furina/onnx/stream150.onnx.json",
          "assets/weights/furina/added_IVF1358_furina.small4.index",
          0.3, 0.5),
}


class RVCPlugin:
    """实时变声插件。

    用法:
        p = RVCPlugin(plan="E", input_device=1, output_device=6)
        p.start()          # 开始变声
        p.set_rms_mix(0.5) # 热调响度因子
        p.stop()           # 停止
    """

    def __init__(self, plan="E", input_device=None, output_device=None,
                 dml_device=0, auto_pitch=True, auto_pitch_mode="continuous",
                 on_output=None, **kw):
        self.plan = plan
        if plan in PLANS:
            onnx, meta, index, index_rate, rms_mix = PLANS[plan]
        else:
            onnx = kw.pop("onnx", "assets/weights/furina/onnx/stream150.int8.onnx")
            meta = kw.pop("meta", "assets/weights/furina/onnx/stream150.onnx.json")
            index = kw.pop("index", None)
            index_rate = kw.pop("index_rate", 0.0)
            rms_mix = kw.pop("rms_mix", None)
        self.engine = RVCOnnxEngine(
            dec_onnx=onnx, dec_meta=meta, index_path=index,
            index_rate=index_rate, rms_mix_rate=rms_mix,
            auto_pitch=auto_pitch, auto_pitch_mode=auto_pitch_mode,
            dml_device=dml_device, **kw)
        self.engine.warmup()
        self.input_device = input_device
        self.output_device = output_device
        self.on_output = on_output
        self._stream = None
        self._stats = []
        self._rtf = 0.0
        self._stop = False

    # ---------- 生命周期 ----------
    def start(self):
        import sounddevice as sd
        engine = self.engine
        block = engine.block_frame
        sr = engine.sr
        cb = self._make_callback(engine, sr, block)
        self._stream = sd.Stream(
            samplerate=sr, blocksize=block, channels=1, dtype="float32",
            device=(self.input_device, self.output_device), callback=cb,
            latency="high")
        self._stream.start()
        return self

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        return self

    # ---------- 热调 ----------
    def set_rms_mix(self, v):
        self.engine.rms_mix_rate = v if v is not None else None

    def set_index_rate(self, v):
        self.engine.index_rate = float(v)

    def set_pitch(self, n):
        self.engine.pitch = int(n)

    @property
    def rtf(self):
        return self._rtf

    # ---------- 内部 ----------
    def _make_callback(self, engine, sr, block):
        plugin = self

        def callback(indata, outdata, frames, _t, status):
            x = indata[:, 0].copy()
            out, dt = engine.process_block(x)
            outdata[:, 0] = out
            plugin._stats.append(dt)
            if len(plugin._stats) >= 100:
                avg = np.mean(plugin._stats)
                plugin._rtf = block / sr / avg
                plugin._stats.clear()
            if plugin.on_output is not None:
                plugin.on_output(out)
        return callback

    def __enter__(self):
        return self.start()

    def __exit__(self, *a):
        self.stop()
