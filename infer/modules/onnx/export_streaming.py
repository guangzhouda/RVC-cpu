import json
import logging
import math
import os
import traceback

import onnx
import onnxruntime
import torch

from infer.lib.jit.get_synthesizer import get_synthesizer

logger = logging.getLogger(__name__)


class StreamingRVCExportWrapper(torch.nn.Module):
    def __init__(
        self,
        net_g,
        if_f0,
        skip_head,
        return_length,
        return_length2,
    ):
        super().__init__()
        self.net_g = net_g
        self.if_f0 = int(if_f0)
        self.skip_head = int(skip_head)
        self.return_length = int(return_length)
        self.return_length2 = int(return_length2)
        self.flow_head = max(self.skip_head - 24, 0)
        self.dec_head = self.skip_head - self.flow_head
        self.register_buffer(
            "flow_head_tensor", torch.LongTensor([self.flow_head]), persistent=False
        )
        self.register_buffer(
            "return_length2_tensor",
            torch.LongTensor([self.return_length2]),
            persistent=False,
        )

    def forward(self, phone, phone_lengths, pitch, pitchf, sid, rnd):
        g = self.net_g.emb_g(sid).unsqueeze(-1)
        if self.if_f0 == 1:
            m_p, logs_p, x_mask = self.net_g.enc_p(
                phone, pitch, phone_lengths, self.flow_head_tensor
            )
        else:
            m_p, logs_p, x_mask = self.net_g.enc_p(
                phone, None, phone_lengths, self.flow_head_tensor
            )
        # NOTE: 0.66666 noise temperature must match infer/lib/infer_pack/models.py
        # SynthesizerTrnMs256NSFsid.infer (line ~766). The caller feeds raw randn;
        # the scaling is baked inside so this is a faithful drop-in for net_g.infer.
        z_p = (m_p + torch.exp(logs_p) * rnd * 0.66666) * x_mask
        z = self.net_g.flow(z_p, x_mask, g=g, reverse=True)
        z = z[:, :, self.dec_head : self.dec_head + self.return_length]
        x_mask = x_mask[:, :, self.dec_head : self.dec_head + self.return_length]
        if self.if_f0 == 1:
            pitchf = pitchf[:, self.skip_head : self.skip_head + self.return_length]
            audio = self.net_g.dec(
                z * x_mask, pitchf, g=g, n_res=self.return_length2_tensor
            )
        else:
            audio = self.net_g.dec(z * x_mask, g=g, n_res=self.return_length2_tensor)
        return audio


def get_streaming_shape(sr, block_time, crossfade_time, extra_time, formant_shift):
    zc = sr // 100
    block_frame = int(np_round(block_time * sr / zc)) * zc
    crossfade_frame = int(np_round(crossfade_time * sr / zc)) * zc
    sola_buffer_frame = min(crossfade_frame, 4 * zc)
    sola_search_frame = zc
    extra_frame = int(np_round(extra_time * sr / zc)) * zc
    phone_length = (
        extra_frame + crossfade_frame + sola_search_frame + block_frame
    ) // zc
    skip_head = extra_frame // zc
    return_length = (block_frame + sola_buffer_frame + sola_search_frame) // zc
    factor = pow(2, formant_shift / 12)
    return_length2 = int(math.ceil(return_length * factor))
    return {
        "zc": int(zc),
        "block_frame": int(block_frame),
        "crossfade_frame": int(crossfade_frame),
        "sola_buffer_frame": int(sola_buffer_frame),
        "sola_search_frame": int(sola_search_frame),
        "extra_frame": int(extra_frame),
        "phone_length": int(phone_length),
        "skip_head": int(skip_head),
        "return_length": int(return_length),
        "return_length2": int(return_length2),
        "factor": float(factor),
    }


def np_round(value):
    return math.floor(value + 0.5)


def build_dummy_inputs(cpt, stream_shape):
    vec_channels = 256 if cpt.get("version", "v1") == "v1" else 768
    inter_channels = int(cpt["config"][2])
    phone_length = int(stream_shape["phone_length"])
    rnd_length = phone_length - max(int(stream_shape["skip_head"]) - 24, 0)
    if rnd_length <= 0:
        raise ValueError("Invalid stream shape: rnd_length <= 0")
    dummy = {
        "phone": torch.rand(1, phone_length, vec_channels),
        "phone_lengths": torch.LongTensor([phone_length]),
        "pitch": torch.randint(low=5, high=255, size=(1, phone_length)),
        "pitchf": torch.rand(1, phone_length),
        "sid": torch.LongTensor([0]),
        "rnd": torch.rand(1, inter_channels, rnd_length),
    }
    return dummy, vec_channels, inter_channels, rnd_length


def save_streaming_metadata(
    metadata_path,
    model_path,
    output_path,
    cpt,
    stream_shape,
    vec_channels,
    inter_channels,
    rnd_length,
):
    metadata = {
        "model_path": os.path.abspath(model_path),
        "onnx_path": os.path.abspath(output_path),
        "version": cpt.get("version", "v1"),
        "if_f0": int(cpt.get("f0", 1)),
        "target_sr": int(cpt["config"][-1]),
        "speaker_count": int(cpt["weight"]["emb_g.weight"].shape[0]),
        "vec_channels": int(vec_channels),
        "inter_channels": int(inter_channels),
        "stream": stream_shape,
        "inputs": {
            "phone": [1, int(stream_shape["phone_length"]), int(vec_channels)],
            "phone_lengths": [1],
            "pitch": [1, int(stream_shape["phone_length"])],
            "pitchf": [1, int(stream_shape["phone_length"])],
            "sid": [1],
            "rnd": [1, int(inter_channels), int(rnd_length)],
        },
        "notes": [
            "This ONNX is exported for the realtime streaming infer path in infer/lib/rtrvc.py.",
            "skip_head, return_length, and return_length2 are baked into the graph for one fixed stream configuration.",
            "C++ inference should feed the exact input tensor shapes recorded in this metadata.",
        ],
    }
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def simplify_onnx(output_path):
    # onnxsim folds constants + runs shape inference, matching the offline
    # export path (infer/modules/onnx/export.py). Optional: if onnxsim is not
    # installed, the raw torch.onnx.export output is still valid, just larger.
    try:
        import onnxsim
    except ImportError:
        logger.warning("onnxsim not installed; skipping simplification")
        return
    model = onnx.load(output_path)
    simplified, check = onnxsim.simplify(model)
    if check:
        onnx.save(simplified, output_path)
        logger.info("onnxsim simplification ok")
    else:
        logger.warning("onnxsim simplify() check failed; keeping raw export")


def verify_onnx(output_path, dummy, if_f0):
    model = onnx.load(output_path)
    onnx.checker.check_model(model)
    session = onnxruntime.InferenceSession(
        output_path, providers=["CPUExecutionProvider"]
    )
    # Feed only the inputs the exported graph actually declares. Tracing drops
    # unused forward args (e.g. pitch/pitchf on the nono path), so assuming the
    # full input_names list would raise "no input found".
    feed = {
        "phone": dummy["phone"].cpu().numpy(),
        "phone_lengths": dummy["phone_lengths"].cpu().numpy(),
        "sid": dummy["sid"].cpu().numpy(),
        "rnd": dummy["rnd"].cpu().numpy(),
    }
    if int(if_f0) == 1:
        feed["pitch"] = dummy["pitch"].cpu().numpy()
        feed["pitchf"] = dummy["pitchf"].cpu().numpy()
    session_input_names = {i.name for i in session.get_inputs()}
    inputs = {k: v for k, v in feed.items() if k in session_input_names}
    missing = session_input_names - set(inputs.keys())
    if missing:
        raise RuntimeError("Missing ONNX inputs: %s" % sorted(missing))
    outputs = session.run(None, inputs)
    if len(outputs) != 1:
        raise RuntimeError("Unexpected ONNX outputs")
    logger.info("onnxruntime output shape: %s", list(outputs[0].shape))


def export_streaming_onnx(
    model_path,
    output_path,
    block_time=0.25,
    crossfade_time=0.05,
    extra_time=2.5,
    formant_shift=0.0,
    verify=True,
):
    try:
        net_g, cpt = get_synthesizer(model_path, torch.device("cpu"))
        stream_shape = get_streaming_shape(
            int(cpt["config"][-1]),
            float(block_time),
            float(crossfade_time),
            float(extra_time),
            float(formant_shift),
        )
        dummy, vec_channels, inter_channels, rnd_length = build_dummy_inputs(
            cpt, stream_shape
        )
        wrapper = StreamingRVCExportWrapper(
            net_g,
            cpt.get("f0", 1),
            stream_shape["skip_head"],
            stream_shape["return_length"],
            stream_shape["return_length2"],
        )
        wrapper.eval()

        input_names = ["phone", "phone_lengths", "pitch", "pitchf", "sid", "rnd"]
        output_names = ["audio"]

        logger.info("Exporting streaming ONNX to %s", output_path)
        logger.info(
            "stream config: skip_head=%s return_length=%s return_length2=%s phone_length=%s",
            stream_shape["skip_head"],
            stream_shape["return_length"],
            stream_shape["return_length2"],
            stream_shape["phone_length"],
        )

        torch.onnx.export(
            wrapper,
            (
                dummy["phone"],
                dummy["phone_lengths"],
                dummy["pitch"],
                dummy["pitchf"],
                dummy["sid"],
                dummy["rnd"],
            ),
            output_path,
            do_constant_folding=False,
            opset_version=18,
            verbose=False,
            input_names=input_names,
            output_names=output_names,
        )

        simplify_onnx(output_path)

        metadata_path = output_path + ".json"
        save_streaming_metadata(
            metadata_path,
            model_path,
            output_path,
            cpt,
            stream_shape,
            vec_channels,
            inter_channels,
            rnd_length,
        )
        logger.info("Saved metadata to %s", metadata_path)

        if verify:
            verify_onnx(output_path, dummy, cpt.get("f0", 1))
        return {
            "status": "ok",
            "onnx_path": os.path.abspath(output_path),
            "metadata_path": os.path.abspath(metadata_path),
            "stream_shape": stream_shape,
        }
    except Exception:
        traceback.print_exc()
        raise
