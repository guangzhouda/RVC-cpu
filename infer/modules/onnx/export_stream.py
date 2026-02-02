import torch
import onnx
import onnxsim

from infer.lib.infer_pack.models import (
    SynthesizerTrnMs256NSFsid,
    SynthesizerTrnMs256NSFsid_nono,
    SynthesizerTrnMs768NSFsid,
    SynthesizerTrnMs768NSFsid_nono,
)


class _InferStreamWrapper(torch.nn.Module):
    """
    说明：
    - 该 wrapper 导出的是“带 skip_head/return_length 裁剪”的推理路径（参考 infer_pack/models.py 的 infer）。
    - 相比普通 forward 导出的 onnx，这个版本可以显著减少解码端/流端的无用计算（CPU-only 更关键）。
    - 为了避免 ONNX 不支持随机数算子，这里把 rnd 作为输入（与本仓库旧 onnx 导出保持一致）。
    """

    def __init__(self, net_g):
        super().__init__()
        self.net_g = net_g

    def forward(self, phone, phone_lengths, pitch, pitchf, sid, rnd, skip_head, return_length):
        # 注意：这里的 skip_head/return_length 是“10ms 帧数”（和 tools/rvc_for_realtime.py 一致）
        # torch.onnx.export 采用 tracing，.item() 会把示例输入的值固化到图里；
        # 因此导出的 onnx 适配固定的 block/extra 配置（运行时也应传入同样的值）。
        head = int(skip_head.item())
        length = int(return_length.item())
        flow_head = torch.clamp(skip_head - 24, min=0)
        dec_head = head - int(flow_head.item())

        g = self.net_g.emb_g(sid).unsqueeze(-1)
        m_p, logs_p, x_mask = self.net_g.enc_p(phone, pitch, phone_lengths, flow_head)
        # rnd 的 shape 需要与 m_p 一致：[B, 192, T]
        z_p = (m_p + torch.exp(logs_p) * rnd) * x_mask
        z = self.net_g.flow(z_p, x_mask, g=g, reverse=True)
        z = z[:, :, dec_head : dec_head + length]
        x_mask = x_mask[:, :, dec_head : dec_head + length]
        pitchf = pitchf[:, head : head + length]
        o = self.net_g.dec(z * x_mask, pitchf, g=g)
        return o


def export_onnx_stream(ModelPath, ExportedPath, skip_head_frames=25, return_length_frames=55):
    """
    导出“流式裁剪”版 synthesizer.onnx（固定 skip_head/return_length）。
    参数单位：10ms 帧（与 realtime 逻辑一致）。
    """

    cpt = torch.load(ModelPath, map_location="cpu")
    cpt["config"][-3] = cpt["weight"]["emb_g.weight"].shape[0]
    if_f0 = cpt.get("f0", 1)
    version = cpt.get("version", "v1")

    synthesizer_class = {
        ("v1", 1): SynthesizerTrnMs256NSFsid,
        ("v1", 0): SynthesizerTrnMs256NSFsid_nono,
        ("v2", 1): SynthesizerTrnMs768NSFsid,
        ("v2", 0): SynthesizerTrnMs768NSFsid_nono,
    }
    net_g = synthesizer_class[(version, if_f0)](*cpt["config"], is_half=False)
    if hasattr(net_g, "enc_q"):
        del net_g.enc_q
    net_g.load_state_dict(cpt["weight"], strict=False)
    net_g.eval().float()

    vec_channels = 256 if version == "v1" else 768
    T = int(skip_head_frames + return_length_frames)
    flow_head_frames = max(int(skip_head_frames) - 24, 0)
    T_flow = max(T - flow_head_frames, 1)

    test_phone = torch.rand(1, T, vec_channels)
    test_phone_lengths = torch.tensor([T]).long()
    test_pitch = torch.randint(size=(1, T), low=1, high=255).long()
    test_pitchf = torch.rand(1, T).float()
    test_sid = torch.LongTensor([0])
    # 注意：infer 路径里 enc_p 会在 flow_head 后裁剪时间轴，因此 rnd 需要与 m_p 的时间长度一致。
    test_rnd = torch.rand(1, 192, T_flow).float()
    test_skip = torch.LongTensor([int(skip_head_frames)])
    test_ret = torch.LongTensor([int(return_length_frames)])

    wrapper = _InferStreamWrapper(net_g)
    input_names = ["phone", "phone_lengths", "pitch", "pitchf", "sid", "rnd", "skip_head", "return_length"]
    output_names = ["audio"]

    torch.onnx.export(
        wrapper,
        (test_phone, test_phone_lengths, test_pitch, test_pitchf, test_sid, test_rnd, test_skip, test_ret),
        ExportedPath,
        dynamic_axes={
            "phone": [1],
            "pitch": [1],
            "pitchf": [1],
            "rnd": [2],
        },
        do_constant_folding=False,
        opset_version=12,
        verbose=False,
        input_names=input_names,
        output_names=output_names,
    )

    model, _ = onnxsim.simplify(ExportedPath)

    # 写入元数据，方便 C++ 侧校验“导出配置”和“运行配置”是否一致
    def _set_meta(k: str, v: int):
        p = model.metadata_props.add()
        p.key = k
        p.value = str(int(v))

    _set_meta("rvc_stream_skip_head_frames", int(skip_head_frames))
    _set_meta("rvc_stream_return_length_frames", int(return_length_frames))
    _set_meta("rvc_stream_flow_head_frames", int(flow_head_frames))
    onnx.save(model, ExportedPath)
    return "Finished"
