# -*- coding: utf-8 -*-
"""
导出 HuBERT 为 ONNX（时间轴动态），供 rvc_onnx_live.py 实时链路使用。

要点:
- 不带 padding_mask（实时路径恒为全 False，等价于 None）；
- v2 模型用 output_layer=12，直接返回 logits[0]；
  若是 v1 模型请改 wrapper 为: logits[0] 经 hubert_model.final_proj 且 output_layer=9；
- 调用方需把 16k 输入补到 640 采样点（2 帧）的整数倍，再按 ceil(T/320) 切片。

用法:
    python tools/export_hubert_onnx.py
"""
import os
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from infer.lib.jit.get_hubert import get_hubert_model  # noqa: E402


def main():
    torch.set_num_threads(1)
    print("loading hubert...")
    hubert = get_hubert_model("assets/hubert/hubert_base.pt", torch.device("cpu")).float().eval()

    class HubertWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, source):
            logits = self.m.extract_features(source=source, padding_mask=None, output_layer=12)
            return logits[0]  # v2 特征（768 维）

    wrap = HubertWrapper(hubert)
    dummy = torch.randn(1, 20000).float()
    out_path = "assets/hubert/hubert_base.onnx"
    torch.onnx.export(
        wrap,
        (dummy,),
        out_path,
        input_names=["source"],
        output_names=["feats"],
        dynamic_axes={"source": {1: "time"}, "feats": {1: "time"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print("saved ->", out_path, "%.1fMB" % (os.path.getsize(out_path) / 1e6))


if __name__ == "__main__":
    main()
