# -*- coding: utf-8 -*-
"""RVC 实时变声 - 插件化 Demo（tkinter，零额外依赖）

用法:
    .venv-onnx-stream\Scripts\python.exe plugin\demo_app.py

功能: 方案切换(E/F) / 设备选择 / 启动停止 / RTF 实时显示 / 响度因子与 index-rate 热调
"""
import os
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from plugin import RVCPlugin  # noqa: E402


class DemoApp:
    def __init__(self, root):
        self.root = root
        self.plugin = None
        root.title("RVC 实时变声插件 Demo (DML)")
        root.geometry("480x420")

        frm = ttk.Frame(root, padding=12)
        frm.pack(fill="both", expand=True)

        # 方案
        ttk.Label(frm, text="方案:").grid(row=0, column=0, sticky="w")
        self.plan_var = tk.StringVar(value="E")
        cb = ttk.Combobox(frm, textvariable=self.plan_var, values=["E", "F"], state="readonly", width=8)
        cb.grid(row=0, column=1, sticky="w")
        ttk.Label(frm, text="E=int8无index  F=fp32+index+rms", foreground="gray").grid(row=0, column=2, sticky="w", padx=8)

        # 设备
        import sounddevice as sd
        devs = sd.query_devices()
        ins = [(i, d["name"]) for i, d in enumerate(devs) if d["max_input_channels"] > 0]
        outs = [(i, d["name"]) for i, d in enumerate(devs) if d["max_output_channels"] > 0]
        ttk.Label(frm, text="输入:").grid(row=1, column=0, sticky="w")
        self.in_var = tk.StringVar(value="1")
        ttk.Combobox(frm, textvariable=self.in_var, values=[str(i) for i, _ in ins], width=8).grid(row=1, column=1, sticky="w")
        ttk.Label(frm, text="输出:").grid(row=2, column=0, sticky="w")
        self.out_var = tk.StringVar(value="6")
        ttk.Combobox(frm, textvariable=self.out_var, values=[str(i) for i, _ in outs], width=8).grid(row=2, column=1, sticky="w")

        # 响度因子 / index rate
        ttk.Label(frm, text="响度因子:").grid(row=3, column=0, sticky="w")
        self.rms_var = tk.DoubleVar(value=0.0)
        rms_s = ttk.Scale(frm, from_=0.0, to=1.0, variable=self.rms_var, command=lambda v: self._apply_rms())
        rms_s.grid(row=3, column=1, columnspan=2, sticky="ew", padx=4)
        self.rms_lbl = ttk.Label(frm, text="0.00")
        self.rms_lbl.grid(row=3, column=3)

        ttk.Label(frm, text="index率:").grid(row=4, column=0, sticky="w")
        self.idx_var = tk.DoubleVar(value=0.0)
        idx_s = ttk.Scale(frm, from_=0.0, to=0.8, variable=self.idx_var, command=lambda v: self._apply_idx())
        idx_s.grid(row=4, column=1, columnspan=2, sticky="ew", padx=4)
        self.idx_lbl = ttk.Label(frm, text="0.00")
        self.idx_lbl.grid(row=4, column=3)

        # 控制
        self.btn = ttk.Button(frm, text="启动变声", command=self.toggle)
        self.btn.grid(row=5, column=0, columnspan=4, pady=10, sticky="ew")

        # 状态
        self.status = ttk.Label(frm, text="未启动")
        self.status.grid(row=6, column=0, columnspan=4)
        self.rtf_lbl = ttk.Label(frm, text="RTF: -")
        self.rtf_lbl.grid(row=7, column=0, columnspan=4)

        self._poll()

    def _apply_rms(self):
        self.rms_lbl.config(text="%.2f" % self.rms_var.get())
        if self.plugin:
            v = self.rms_var.get()
            self.plugin.set_rms_mix(v if v > 0.001 else None)

    def _apply_idx(self):
        self.idx_lbl.config(text="%.2f" % self.idx_var.get())
        if self.plugin:
            self.plugin.set_index_rate(self.idx_var.get())

    def toggle(self):
        if self.plugin is None:
            plan = self.plan_var.get()
            self.status.config(text="加载 %s 方案中..." % plan)
            self.root.update()
            try:
                self.plugin = RVCPlugin(
                    plan=plan,
                    input_device=int(self.in_var.get()),
                    output_device=int(self.out_var.get()),
                    dml_device=0,
                ).start()
            except Exception as e:
                self.status.config(text="启动失败: %s" % e)
                self.plugin = None
                return
            self.btn.config(text="停止")
            self.status.config(text="运行中 (%s)" % plan)
            self._apply_rms(); self._apply_idx()
        else:
            self.plugin.stop()
            self.plugin = None
            self.btn.config(text="启动变声")
            self.status.config(text="已停止")

    def _poll(self):
        if self.plugin is not None:
            self.rtf_lbl.config(text="RTF: %.2fx (实时=1x)" % self.plugin.rtf if self.plugin.rtf else "RTF: 计算中...")
        self.root.after(500, self._poll)


def main():
    root = tk.Tk()
    DemoApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
