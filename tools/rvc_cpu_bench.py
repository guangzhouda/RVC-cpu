"""
RVC PyTorch 实时推理离线基准 — 复刻 gui_v1.py 的滑动缓冲 + SOLA 流程，不开声卡。

用法（在仓库根目录）:
    python tools/rvc_cpu_bench.py --wav input.wav
    python tools/rvc_cpu_bench.py --wav input.wav --device cuda
    python tools/rvc_cpu_bench.py --wav input.wav --quantize --extra-time 1.0 --threads 8

逐块打印各阶段耗时，最后输出拼接好的 wav（默认 rvc_cpu_bench_out.wav），
可直接试听 SOLA 拼接质量。realtime 判定: 每块总耗时 < block_time。
"""
import argparse
import json
import os
import sys
import time

now_dir = os.getcwd()
sys.path.append(now_dir)

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio.transforms as tat
from torchaudio.transforms import Resample


def deterministic_rvc_infer(
    rvc,
    input_wav,
    input_wav_res,
    block_frame_16k,
    skip_head,
    return_length,
    f0method,
    rng,
    debug_finite=False,
):
    def debug_stats(name, tensor):
        if not debug_finite:
            return
        t = tensor.detach().float().cpu()
        print(
            "%s finite=%s min=%.6f max=%.6f mean=%.6f"
            % (
                name,
                bool(torch.isfinite(t).all().item()),
                float(torch.nan_to_num(t).min().item()),
                float(torch.nan_to_num(t).max().item()),
                float(torch.nan_to_num(t).mean().item()),
            )
        )

    t1 = time.perf_counter()
    with torch.no_grad():
        if rvc.config.is_half:
            feats = input_wav_res.half().view(1, -1)
        else:
            feats = input_wav_res.float().view(1, -1)
        padding_mask = torch.BoolTensor(feats.shape).to(rvc.device).fill_(False)
        inputs = {
            "source": feats,
            "padding_mask": padding_mask,
            "output_layer": 9 if rvc.version == "v1" else 12,
        }
        logits = rvc.model.extract_features(**inputs)
        feats = (
            rvc.model.final_proj(logits[0]) if rvc.version == "v1" else logits[0]
        )
        feats = torch.cat((feats, feats[:, -1:, :]), 1)
        debug_stats("hubert_feats", feats)
    t2 = time.perf_counter()

    try:
        if hasattr(rvc, "index") and rvc.index_rate != 0:
            npy = feats[0][skip_head // 2 :].float().cpu().numpy().astype("float32")
            score, ix = rvc.index.search(npy, k=8)
            if (ix >= 0).all():
                weight = np.square(1 / score)
                weight /= weight.sum(axis=1, keepdims=True)
                npy = np.sum(
                    rvc.big_npy[ix] * np.expand_dims(weight, axis=2), axis=1
                )
                if rvc.config.is_half:
                    npy = npy.astype("float16")
                feats[0][skip_head // 2 :] = (
                    torch.from_numpy(npy).unsqueeze(0).to(rvc.device)
                    * rvc.index_rate
                    + (1 - rvc.index_rate) * feats[0][skip_head // 2 :]
                )
            else:
                print("Invalid index. You MUST use added_xxxx.index but not trained_xxxx.index!")
        else:
            print("Index search FAILED or disabled")
    except Exception:
        import traceback

        traceback.print_exc()
        print("Index search FAILED")
    t3 = time.perf_counter()

    p_len = input_wav_res.shape[0] // 160
    factor = pow(2, rvc.formant_shift / 12)
    return_length2 = int(np.ceil(return_length * factor))

    if rvc.if_f0 == 1:
        f0_extractor_frame = block_frame_16k + 800
        if f0method == "rmvpe":
            f0_extractor_frame = 5120 * ((f0_extractor_frame - 1) // 5120 + 1) - 160
        pitch, pitchf = rvc.get_f0(
            input_wav[-f0_extractor_frame:],
            rvc.f0_up_key - rvc.formant_shift,
            rvc.n_cpu,
            f0method,
        )
        shift = block_frame_16k // 160
        rvc.cache_pitch[:-shift] = rvc.cache_pitch[shift:].clone()
        rvc.cache_pitchf[:-shift] = rvc.cache_pitchf[shift:].clone()
        rvc.cache_pitch[4 - pitch.shape[0] :] = pitch[3:-1]
        rvc.cache_pitchf[4 - pitch.shape[0] :] = pitchf[3:-1]
        cache_pitch = rvc.cache_pitch[None, -p_len:]
        cache_pitchf = rvc.cache_pitchf[None, -p_len:] * return_length2 / return_length
    else:
        cache_pitch = None
        cache_pitchf = None
    t4 = time.perf_counter()

    feats = F.interpolate(feats.permute(0, 2, 1), scale_factor=2).permute(0, 2, 1)
    feats = feats[:, :p_len, :]
    debug_stats("interp_feats", feats)

    p_len_tensor = torch.LongTensor([p_len]).to(rvc.device)
    sid = torch.LongTensor([0]).to(rvc.device)
    skip_head_tensor = torch.LongTensor([skip_head]).to(rvc.device)
    return_length_tensor = torch.LongTensor([return_length]).to(rvc.device)
    return_length2_tensor = torch.LongTensor([return_length2]).to(rvc.device)

    with torch.no_grad():
        g = rvc.net_g.emb_g(sid).unsqueeze(-1)
        head = skip_head
        length = return_length
        flow_head = max(skip_head - 24, 0)
        dec_head = head - flow_head

        if rvc.if_f0 == 1:
            m_p, logs_p, x_mask = rvc.net_g.enc_p(
                feats, cache_pitch, p_len_tensor, torch.LongTensor([flow_head]).to(rvc.device)
            )
        else:
            m_p, logs_p, x_mask = rvc.net_g.enc_p(
                feats, None, p_len_tensor, torch.LongTensor([flow_head]).to(rvc.device)
            )
        debug_stats("m_p", m_p)
        debug_stats("logs_p", logs_p)
        debug_stats("x_mask", x_mask)

        rnd = rng.standard_normal(m_p.shape, dtype=np.float32)
        rnd = torch.from_numpy(rnd).to(rvc.device, dtype=m_p.dtype)
        z_p = (m_p + torch.exp(logs_p) * rnd * 0.66666) * x_mask
        debug_stats("z_p", z_p)
        z = rvc.net_g.flow(z_p, x_mask, g=g, reverse=True)
        debug_stats("z", z)
        z = z[:, :, dec_head : dec_head + length]
        x_mask = x_mask[:, :, dec_head : dec_head + length]
        if rvc.if_f0 == 1:
            cache_pitchf = cache_pitchf[:, head : head + length]
            debug_stats("cache_pitchf", cache_pitchf)
            infered_audio = rvc.net_g.dec(
                z * x_mask, cache_pitchf, g=g, n_res=return_length2_tensor
            )
        else:
            infered_audio = rvc.net_g.dec(
                z * x_mask, g=g, n_res=return_length2_tensor
            )
        debug_stats("infered_audio_raw", infered_audio)

    infered_audio = infered_audio.squeeze(1).float()
    upp_res = int(np.floor(factor * rvc.tgt_sr // 100))
    if upp_res != rvc.tgt_sr // 100:
        if upp_res not in rvc.resample_kernel:
            rvc.resample_kernel[upp_res] = Resample(
                orig_freq=upp_res,
                new_freq=rvc.tgt_sr // 100,
                dtype=torch.float32,
            ).to(rvc.device)
        infered_audio = rvc.resample_kernel[upp_res](
            infered_audio[:, : return_length * upp_res]
        )
    t5 = time.perf_counter()
    print(
        "Spent time: fea = %.3fs, index = %.3fs, f0 = %.3fs, model = %.3fs"
        % (t2 - t1, t3 - t2, t4 - t3, t5 - t4)
    )
    return infered_audio.squeeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pth", default="assets/weights/hatsune_miku_model.pth")
    parser.add_argument("--index", default="assets/weights/hatsune_miku_model.index")
    parser.add_argument("--index-rate", type=float, default=0.0)
    parser.add_argument("--wav", required=True, help="输入音频")
    parser.add_argument("--out", default="rvc_cpu_bench_out.wav")
    parser.add_argument("--pitch", type=int, default=0)
    parser.add_argument("--block-time", type=float, default=0.25)
    parser.add_argument("--crossfade-time", type=float, default=0.05)
    parser.add_argument("--extra-time", type=float, default=2.5)
    parser.add_argument("--f0method", default="fcpe")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="cpu=RVC CPU benchmark, cuda=RVC GPU realtime baseline",
    )
    parser.add_argument("--quantize", action="store_true",
                        help="对 Hubert/net_g 的 Linear 层做 int8 动态量化")
    parser.add_argument("--max-blocks", type=int, default=0, help="0=整个文件")
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="固定潜变量噪声，便于 CPU/GPU 与 baseline 做客观比较",
    )
    parser.add_argument(
        "--report",
        default="",
        help="可选 JSON 报告输出路径",
    )
    parser.add_argument(
        "--debug-finite",
        action="store_true",
        help="打印关键张量的 finite/min/max 诊断信息",
    )
    args, _ = parser.parse_known_args()

    effective_threads = args.threads
    if args.device == "cpu" and args.threads > 1:
        print(
            "warning: multi-threaded CPU HuBERT becomes non-finite on this machine; "
            "falling back to threads=1 for a valid RVC CPU benchmark"
        )
        effective_threads = 1

    torch.set_num_threads(effective_threads)
    sys.argv = [sys.argv[0]]  # 防止 Config 解析到我们的参数

    from configs.config import Config
    from infer.lib import rtrvc as rvc_for_realtime

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if args.device == "cuda" and args.quantize:
        raise RuntimeError("Dynamic quantization is only supported for CPU benchmark")

    config = Config()
    config.device = torch.device("cuda:0" if args.device == "cuda" else "cpu")
    config.is_half = args.device == "cuda"
    config.use_jit = False

    rvc = rvc_for_realtime.RVC(
        args.pitch, 0, args.pth, args.index, args.index_rate,
        1, None, None, config, None,
    )

    if args.quantize:
        rvc.model = torch.ao.quantization.quantize_dynamic(
            rvc.model, {torch.nn.Linear}, dtype=torch.qint8
        )
        rvc.net_g = torch.ao.quantization.quantize_dynamic(
            rvc.net_g, {torch.nn.Linear}, dtype=torch.qint8
        )
        print("dynamic int8 quantization applied")

    samplerate = rvc.tgt_sr
    zc = samplerate // 100
    block_frame = int(np.round(args.block_time * samplerate / zc)) * zc
    block_frame_16k = 160 * block_frame // zc
    crossfade_frame = int(np.round(args.crossfade_time * samplerate / zc)) * zc
    sola_buffer_frame = min(crossfade_frame, 4 * zc)
    sola_search_frame = zc
    extra_frame = int(np.round(args.extra_time * samplerate / zc)) * zc
    device = config.device

    input_wav = torch.zeros(
        extra_frame + crossfade_frame + sola_search_frame + block_frame,
        dtype=torch.float32,
        device=device,
    )
    input_wav_res = torch.zeros(
        160 * input_wav.shape[0] // zc, dtype=torch.float32, device=device
    )
    sola_buffer = torch.zeros(sola_buffer_frame, dtype=torch.float32, device=device)
    fade_in_window = (
        torch.sin(
            0.5
            * np.pi
            * torch.linspace(0.0, 1.0, steps=sola_buffer_frame, device=device)
        )
        ** 2
    )
    fade_out_window = 1 - fade_in_window
    skip_head = extra_frame // zc
    return_length = (block_frame + sola_buffer_frame + sola_search_frame) // zc
    resampler = tat.Resample(
        orig_freq=samplerate, new_freq=16000, dtype=torch.float32
    ).to(device)

    import soundfile as sf
    audio, in_sr = sf.read(args.wav)
    if audio.ndim > 1:
        audio = audio.mean(-1)
    audio = torch.from_numpy(audio.astype(np.float32))
    if in_sr != samplerate:
        audio = tat.Resample(in_sr, samplerate, dtype=torch.float32)(audio)
    audio = audio.to(device)
    n_blocks = len(audio) // block_frame
    if args.max_blocks:
        n_blocks = min(n_blocks, args.max_blocks)
    rng = np.random.default_rng(args.seed)
    print(
        f"device={args.device} tgt_sr={samplerate} block={block_frame} ({args.block_time*1000:.0f}ms) "
        f"extra={args.extra_time}s crossfade={args.crossfade_time}s "
        f"f0={args.f0method} threads={effective_threads} blocks={n_blocks}"
    )

    out_blocks = []
    times = []
    sola_offsets = []
    for i in range(n_blocks):
        indata = audio[i * block_frame: (i + 1) * block_frame]
        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        # --- 滑动输入缓冲 (同 gui_v1.audio_callback) ---
        input_wav[:-block_frame] = input_wav[block_frame:].clone()
        input_wav[-block_frame:] = indata
        input_wav_res[:-block_frame_16k] = input_wav_res[block_frame_16k:].clone()
        input_wav_res[-160 * (block_frame // zc + 1):] = resampler(
            input_wav[-block_frame - 2 * zc:]
        )[160:]
        # --- 推理 ---
        infer_wav = deterministic_rvc_infer(
            rvc,
            input_wav,
            input_wav_res,
            block_frame_16k,
            skip_head,
            return_length,
            args.f0method,
            rng,
            debug_finite=args.debug_finite,
        )
        if args.device == "cuda":
            torch.cuda.synchronize()
        # --- SOLA ---
        conv_input = infer_wav[None, None, : sola_buffer_frame + sola_search_frame]
        cor_nom = F.conv1d(conv_input, sola_buffer[None, None, :])
        cor_den = torch.sqrt(
            F.conv1d(
                conv_input ** 2,
                torch.ones(1, 1, sola_buffer_frame, device=device),
            )
            + 1e-8
        )
        sola_offset = int(torch.argmax(cor_nom[0, 0] / cor_den[0, 0]))
        sola_offsets.append(sola_offset)
        infer_wav = infer_wav[sola_offset:]
        infer_wav[:sola_buffer_frame] *= fade_in_window
        infer_wav[:sola_buffer_frame] += sola_buffer * fade_out_window
        sola_buffer[:] = infer_wav[block_frame: block_frame + sola_buffer_frame]
        out_blocks.append(infer_wav[:block_frame].float().cpu().clone().numpy())
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"block {i}: {dt*1000:.0f}ms sola_offset={sola_offset}")

    times = np.array(times[1:]) if len(times) > 1 else np.array(times)
    budget = args.block_time
    avg_ms = float(times.mean() * 1000)
    p95_ms = float(np.percentile(times, 95) * 1000)
    realtime_ok = bool(np.percentile(times, 95) < budget)
    rtf = float(budget / times.mean())
    print(
        f"\navg={avg_ms:.0f}ms  p95={p95_ms:.0f}ms "
        f"/ budget={budget*1000:.0f}ms  -> "
        f"{'REALTIME OK' if realtime_ok else 'TOO SLOW'} "
        f"(RTF={rtf:.2f}x)"
    )
    print(
        f"sola_offset: mean={np.mean(sola_offsets):.1f} max={max(sola_offsets)} "
        f"(search range 0~{sola_search_frame})"
    )
    sf.write(args.out, np.concatenate(out_blocks), samplerate, subtype="FLOAT")
    print(f"saved -> {args.out}")
    if args.report:
        report = {
            "engine": "rvc_pytorch",
            "device": args.device,
            "wav": os.path.abspath(args.wav),
            "out": os.path.abspath(args.out),
            "sample_rate": int(samplerate),
            "block_time": float(args.block_time),
            "crossfade_time": float(args.crossfade_time),
            "extra_time": float(args.extra_time),
            "f0method": args.f0method,
            "seed": int(args.seed),
            "threads_requested": int(args.threads),
            "threads_effective": int(effective_threads),
            "max_blocks": int(args.max_blocks),
            "blocks": int(n_blocks),
            "avg_ms": avg_ms,
            "p95_ms": p95_ms,
            "rtf": rtf,
            "realtime_ok": realtime_ok,
            "sola_offset_mean": float(np.mean(sola_offsets)),
            "sola_offset_max": int(max(sola_offsets)),
        }
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"report -> {args.report}")


if __name__ == "__main__":
    main()
