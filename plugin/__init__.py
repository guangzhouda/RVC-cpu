# -*- coding: utf-8 -*-
"""
rvc_plugin - RVC 实时变声插件模块（DML 推理）
================================================
把 apps/live_onnx.py 的实时引擎封装成可复用插件：
    - start()/stop(): 声卡流控制
    - set_*(): 运行中热调参数
    - 回调：on_output(wav_block) 可挂接录音/转发/虚拟声卡

依赖（最小集）:
    python 3.10 + onnxruntime-directml + numpy + sounddevice
    + torch/torchaudio（重采样/SOLA；后续可去 torch 化）
"""
from .engine import RVCPlugin

__all__ = ["RVCPlugin"]
__version__ = "0.1.0"
