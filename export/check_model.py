# -*- coding: utf-8 -*-
"""检查 RVC 模型信息：版本/采样率/说话人数/是否带 f0"""
import sys
import torch

path = sys.argv[1]
cpt = torch.load(path, map_location="cpu")
cfg = cpt.get("config", [])
print("keys:", list(cpt.keys())[:6])
print("version:", cpt.get("version", "?"))
print("sr:", cpt.get("sr", "?"), "| f0:", cpt.get("f0", "?"))
print("spk_count (config[-3]):", cfg[-3] if len(cfg) >= 3 else "N/A")
print("sr_from_config (config[-1]):", cfg[-1] if len(cfg) >= 1 else "N/A")
