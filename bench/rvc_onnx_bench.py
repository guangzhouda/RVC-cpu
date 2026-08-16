"""
RVC ONNX 流式实时基准 — 复刻 gui_v1.py / rvc_cpu_bench.py 的滑动缓冲 + SOLA 流程，
仅将最后的声码器/解码阶段替换为流式 ONNX。

用途:
1. 用命令行观察流式 ONNX 的逐块延迟和实时性
2. 输出拼接后的 wav，直接试听实际效果

示例:
    .\.venv-onnx-stream\Scripts\python.exe tools\rvc_onnx_bench.py --wav input.wav
    .\.venv-onnx-stream\Scripts\python.exe tools\rvc_onnx_bench.py --wav input.wav --provider cpu
"""
import argparse
import json
import os
import sys
import time

now_dir = os.getcwd()
sys.path.append(now_dir)

import numpy as np
import onnxruntime
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat


def load_metadata(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_provider_list(provider):
    provider = provider.lower()
    if provider == "cpu":
        return ["CPUExecutionProvider"]
    if provider == "cuda":
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    if provider == "dml":
        return ["DmlExecutionProvider", "CPUExecutionProvider"]
    raise ValueError("provider must be one of: cpu, cuda, dml")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", default="assets/weights/miku/onnx/stream150.onnx")
    parser.add_argument("--meta", default="assets/weights/miku/onnx/stream150.onnx.json")
    parser.add_argument("--pth", default="assets/weights/miku/hatsune_miku_model.pth")
    parser.add_argument("--index", default="assets/weights/miku/hatsune_miku_model.index")
    parser.add_argument("--index-rate", type=float, default=0.0)
    parser.add_argument("--wav", required=True, help="输入音频")
    parser.add_argument("--out", default="rvc_onnx_bench_out.wav")
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--provider", default="cpu")
    parser.add_argument("--f0method", default="fcpe")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--max-blocks", type=int, default=0, help="0=整个文件")
    parser.add_argument("--seed", type=int, default=1234)
    args, _ = parser.parse_known_args()

    metadata = load_metadata(args.meta)
    stream = metadata["stream"]
    inputs = metadata["inputs"]

    torch.set_num_threads(args.threads)
    sys.argv = [sys.argv[0]]

    from configs.config import Config
    from infer.lib import rtrvc as rvc_for_realtime

    config = Config()
    config.device = torch.device("cpu")
    config.is_half = False
    config.use_jit = False

    rvc = rvc_for_realtime.RVC(
        args.pitch,
        0,
        args.pth,
        args.index,
        args.index_rate,
        1,
        None,
        None,
        config,
        None,
    )

    samplerate = rvc.tgt_sr
    if samplerate != int(metadata["target_sr"]):
        raise RuntimeError("target_sr mismatch between pth and onnx metadata")

    zc = int(stream["zc"])
    block_frame = int(stream["block_frame"])
    block_frame_16k = 160 * block_frame // zc
    crossfade_frame = int(stream["crossfade_frame"])
    sola_buffer_frame = int(stream["sola_buffer_frame"])
    sola_search_frame = int(stream["sola_search_frame"])
    extra_frame = int(stream["extra_frame"])
    skip_head = int(stream["skip_head"])
    return_length = int(stream["return_length"])
    phone_length = int(stream["phone_length"])
    rnd_shape = [int(v) for v in inputs["rnd"]]

    input_wav = torch.zeros(
        extra_frame + crossfade_frame + sola_search_frame + block_frame,
        dtype=torch.float32,
    )
    input_wav_res = torch.zeros(160 * input_wav.shape[0] // zc, dtype=torch.float32)
    sola_buffer = torch.zeros(sola_buffer_frame, dtype=torch.float32)
    fade_in_window = (
        torch.sin(0.5 * np.pi * torch.linspace(0.0, 1.0, steps=sola_buffer_frame)) ** 2
    )
    fade_out_window = 1 - fade_in_window
    resampler = tat.Resample(orig_freq=samplerate, new_freq=16000, dtype=torch.float32)

    providers = get_provider_list(args.provider)
    session = onnxruntime.InferenceSession(args.onnx, providers=providers)
    print("providers:", session.get_providers())

    audio, in_sr = sf.read(args.wav)
    if audio.ndim > 1:
        audio = audio.mean(-1)
    audio = torch.from_numpy(audio.astype(np.float32))
    if in_sr != samplerate:
        audio = tat.Resample(in_sr, samplerate, dtype=torch.float32)(audio)

    n_blocks = len(audio) // block_frame
    if args.max_blocks:
        n_blocks = min(n_blocks, args.max_blocks)

    print(
        f"tgt_sr={samplerate} block={block_frame} ({block_frame / samplerate * 1000:.0f}ms) "
        f"extra={extra_frame / samplerate:.2f}s crossfade={crossfade_frame / samplerate:.2f}s "
        f"f0={args.f0method} provider={args.provider} threads={args.threads} blocks={n_blocks}"
    )

    rng = np.random.default_rng(args.seed)
    out_blocks = []
    total_times = []
    dec_times = []
    sola_offsets = []

    for i in range(n_blocks):
        indata = audio[i * block_frame : (i + 1) * block_frame]
        t0 = time.perf_counter()

        input_wav[:-block_frame] = input_wav[block_frame:].clone()
        input_wav[-block_frame:] = indata
        input_wav_res[:-block_frame_16k] = input_wav_res[block_frame_16k:].clone()
        input_wav_res[-160 * (block_frame // zc + 1) :] = resampler(
            input_wav[-block_frame - 2 * zc :]
        )[160:]

        with torch.no_grad():
            feats = input_wav_res.float().view(1, -1)
            padding_mask = torch.BoolTensor(feats.shape).fill_(False)
            model_inputs = {
                "source": feats,
                "padding_mask": padding_mask,
                "output_layer": 9 if rvc.version == "v1" else 12,
            }
            logits = rvc.model.extract_features(**model_inputs)
            feats = rvc.model.final_proj(logits[0]) if rvc.version == "v1" else logits[0]
            feats = torch.cat((feats, feats[:, -1:, :]), 1)

        if hasattr(rvc, "index") and rvc.index_rate != 0:
            npy = feats[0][skip_head // 2 :].cpu().numpy().astype("float32")
            score, ix = rvc.index.search(npy, k=8)
            if (ix >= 0).all():
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(rvc.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1)
                feats[0][skip_head // 2 :] = (
                    torch.from_numpy(npy).unsqueeze(0) * rvc.index_rate
                    + (1 - rvc.index_rate) * feats[0][skip_head // 2 :]
                )

        p_len = input_wav_res.shape[0] // 160
        if p_len != phone_length:
            raise RuntimeError(
                f"phone_length mismatch: frontend={p_len}, onnx={phone_length}. "
                "You must benchmark with the same block/crossfade/extra settings used at export time."
            )

        f0_extractor_frame = block_frame_16k + 800
        if args.f0method == "rmvpe":
            f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
        pitch, pitchf = rvc.get_f0(
            input_wav[-f0_extractor_frame :], rvc.f0_up_key - rvc.formant_shift, 1, args.f0method
        )
        shift = block_frame_16k // 160
        rvc.cache_pitch[:-shift] = rvc.cache_pitch[shift:].clone()
        rvc.cache_pitchf[:-shift] = rvc.cache_pitchf[shift:].clone()
        rvc.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
        rvc.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
        cache_pitch = rvc.cache_pitch[None, -p_len:]
        cache_pitchf = rvc.cache_pitchf[None, -p_len:]

        feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
        feats = feats[:, :p_len, :]
        phone_lengths = np.array([p_len], dtype=np.int64)
        sid = np.array([0], dtype=np.int64)
        rnd = rng.standard_normal(rnd_shape, dtype=np.float32)

        onnx_input = {
            "phone": feats.cpu().numpy().astype(np.float32),
            "phone_lengths": phone_lengths,
            "pitch": cache_pitch.cpu().numpy().astype(np.int64),
            "pitchf": cache_pitchf.cpu().numpy().astype(np.float32),
            "sid": sid,
            "rnd": rnd,
        }

        t_dec0 = time.perf_counter()
        infer_wav = session.run(None, onnx_input)[0]
        t_dec1 = time.perf_counter()
        dec_times.append(t_dec1 - t_dec0)

        infer_wav = torch.from_numpy(infer_wav).squeeze(0).squeeze(0).float()

        conv_input = infer_wav[None, None, : sola_buffer_frame + sola_search_frame]
        cor_nom = F.conv1d(conv_input, sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(conv_input ** 2, torch.ones(1, 1, sola_buffer_frame)) + 1e-8
        )
        sola_offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]))
        sola_offsets.append(sola_offset)
        infer_wav = infer_wav[sola_offset:]
        infer_wav[:sola_buffer_frame] *= fade_in_window
        infer_wav[:sola_buffer_frame] += sola_buffer * fade_out_window
        sola_buffer[:] = infer_wav[block_frame : block_frame + sola_buffer_frame]
        out_blocks.append(infer_wav[:block_frame].clone().numpy())

        dt = time.perf_counter() - t0
        total_times.append(dt)
        print(
            f"block {i}: total={dt*1000:.0f}ms onnx={dec_times[-1]*1000:.0f}ms sola_offset={sola_offset}"
        )

    total_times = np.array(total_times[1:]) if len(total_times) > 1 else np.array(total_times)
    dec_times = np.array(dec_times[1:]) if len(dec_times) > 1 else np.array(dec_times)
    budget = block_frame / samplerate

    print(
        f"\nonnx avg={dec_times.mean()*1000:.0f}ms p95={np.percentile(dec_times, 95)*1000:.0f}ms"
    )
    print(
        f"total avg={total_times.mean()*1000:.0f}ms p95={np.percentile(total_times, 95)*1000:.0f}ms "
        f"/ budget={budget*1000:.0f}ms -> "
        f"{'REALTIME OK' if np.percentile(total_times, 95) < budget else 'TOO SLOW'} "
        f"(RTF={budget / total_times.mean():.2f}x)"
    )
    print(
        f"sola_offset: mean={np.mean(sola_offsets):.1f} max={max(sola_offsets)} "
        f"(search range 0~{sola_search_frame})"
    )

    sf.write(args.out, np.concatenate(out_blocks), samplerate, subtype="FLOAT")
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
