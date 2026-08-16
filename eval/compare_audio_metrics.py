"""
对比两段音频的客观指标，默认把 baseline 作为参考。

指标:
- 对齐延迟 (lag)
- waveform MAE / RMSE / SI-SDR
- log-mel L1 / L2
- MCD (基于 MFCC 1..13)
- F0 RMSE / voiced ratio
"""
import argparse
import json
import os

import librosa
import numpy as np
import scipy.signal
import soundfile as sf


def load_audio(path, sr):
    audio, in_sr = sf.read(path)
    if audio.ndim > 1:
        audio = audio.mean(-1)
    audio = audio.astype(np.float32)
    if in_sr != sr:
        audio = librosa.resample(audio, orig_sr=in_sr, target_sr=sr)
    return audio


def align_audio(reference, candidate, max_shift_samples):
    corr = scipy.signal.correlate(candidate, reference, mode="full", method="fft")
    lags = scipy.signal.correlation_lags(len(candidate), len(reference), mode="full")
    mask = np.abs(lags) <= max_shift_samples
    corr = corr[mask]
    lags = lags[mask]
    lag = int(lags[np.argmax(np.abs(corr))])

    if lag >= 0:
        cand = candidate[lag:]
        ref = reference[: len(cand)]
    else:
        ref = reference[-lag:]
        cand = candidate[: len(ref)]

    length = min(len(ref), len(cand))
    ref = ref[:length]
    cand = cand[:length]
    return ref, cand, lag


def calc_sisdr(reference, estimate):
    eps = 1e-8
    ref = reference.astype(np.float64)
    est = estimate.astype(np.float64)
    scale = np.dot(est, ref) / (np.dot(ref, ref) + eps)
    proj = scale * ref
    noise = est - proj
    return float(
        10 * np.log10((np.sum(proj ** 2) + eps) / (np.sum(noise ** 2) + eps))
    )


def calc_logmel(audio, sr):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        n_mels=80,
        power=2.0,
    )
    mel = librosa.power_to_db(np.maximum(mel, 1e-10), ref=1.0)
    return mel


def calc_mfcc(audio, sr):
    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sr,
        n_mfcc=13,
        n_fft=1024,
        hop_length=256,
        win_length=1024,
    )
    return mfcc


def calc_mcd(ref_mfcc, cand_mfcc):
    length = min(ref_mfcc.shape[1], cand_mfcc.shape[1])
    if length == 0:
        return None
    diff = ref_mfcc[:, :length] - cand_mfcc[:, :length]
    dist = np.sqrt(np.sum(diff ** 2, axis=0))
    return float((10.0 / np.log(10.0)) * np.sqrt(2.0) * np.mean(dist))


def calc_f0_pyin(audio, sr):
    f0, voiced_flag, _ = librosa.pyin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=1024,
        hop_length=256,
    )
    return f0, voiced_flag


def calc_f0_yin(audio, sr):
    return librosa.yin(
        audio,
        fmin=librosa.note_to_hz("C2"),
        fmax=librosa.note_to_hz("C7"),
        sr=sr,
        frame_length=1024,
        hop_length=256,
    )


def compare_metrics(reference, candidate, sr):
    wave_mae = float(np.mean(np.abs(reference - candidate)))
    wave_rmse = float(np.sqrt(np.mean((reference - candidate) ** 2)))
    sisdr = calc_sisdr(reference, candidate)

    ref_mel = calc_logmel(reference, sr)
    cand_mel = calc_logmel(candidate, sr)
    mel_len = min(ref_mel.shape[1], cand_mel.shape[1])
    mel_diff = ref_mel[:, :mel_len] - cand_mel[:, :mel_len]
    logmel_l1 = float(np.mean(np.abs(mel_diff)))
    logmel_l2 = float(np.sqrt(np.mean(mel_diff ** 2)))

    ref_mfcc = calc_mfcc(reference, sr)
    cand_mfcc = calc_mfcc(candidate, sr)
    mcd = calc_mcd(ref_mfcc, cand_mfcc)

    ref_f0, ref_voiced = calc_f0_pyin(reference, sr)
    cand_f0, cand_voiced = calc_f0_pyin(candidate, sr)
    f0_len = min(len(ref_f0), len(cand_f0))
    both_voiced = ref_voiced[:f0_len] & cand_voiced[:f0_len]

    ref_f0_yin = calc_f0_yin(reference, sr)
    cand_f0_yin = calc_f0_yin(candidate, sr)
    ref_rms = librosa.feature.rms(y=reference, frame_length=1024, hop_length=256)[0]
    cand_rms = librosa.feature.rms(y=candidate, frame_length=1024, hop_length=256)[0]
    yin_len = min(len(ref_f0_yin), len(cand_f0_yin), len(ref_rms), len(cand_rms))
    voiced_mask_yin = (
        np.isfinite(ref_f0_yin[:yin_len])
        & np.isfinite(cand_f0_yin[:yin_len])
        & (ref_rms[:yin_len] > 1e-4)
        & (cand_rms[:yin_len] > 1e-4)
    )
    if np.any(voiced_mask_yin):
        f0_rmse_hz = float(
            np.sqrt(
                np.mean(
                    (ref_f0_yin[:yin_len][voiced_mask_yin] - cand_f0_yin[:yin_len][voiced_mask_yin]) ** 2
                )
            )
        )
        f0_corr = float(
            np.corrcoef(
                ref_f0_yin[:yin_len][voiced_mask_yin],
                cand_f0_yin[:yin_len][voiced_mask_yin],
            )[0, 1]
        )
    else:
        f0_rmse_hz = None
        f0_corr = None

    return {
        "wave_mae": wave_mae,
        "wave_rmse": wave_rmse,
        "si_sdr_db": sisdr,
        "logmel_l1": logmel_l1,
        "logmel_l2": logmel_l2,
        "mcd": mcd,
        "f0_rmse_hz": f0_rmse_hz,
        "f0_corr": f0_corr,
        "pyin_voiced_ratio_reference": float(np.mean(ref_voiced)),
        "pyin_voiced_ratio_candidate": float(np.mean(cand_voiced)),
        "pyin_voiced_overlap_ratio": float(np.mean(both_voiced)) if f0_len else 0.0,
        "yin_voiced_overlap_ratio": float(np.mean(voiced_mask_yin)) if yin_len else 0.0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--candidate", required=True)
    parser.add_argument("--report", default="")
    parser.add_argument("--sr", type=int, default=16000)
    parser.add_argument("--max-shift-ms", type=float, default=1000.0)
    args = parser.parse_args()

    baseline = load_audio(args.baseline, args.sr)
    candidate = load_audio(args.candidate, args.sr)
    max_shift_samples = int(args.max_shift_ms * args.sr / 1000.0)
    ref, cand, lag = align_audio(baseline, candidate, max_shift_samples)
    metrics = compare_metrics(ref, cand, args.sr)

    report = {
        "baseline": os.path.abspath(args.baseline),
        "candidate": os.path.abspath(args.candidate),
        "sample_rate": int(args.sr),
        "lag_samples": int(lag),
        "lag_ms": float(lag * 1000.0 / args.sr),
        "aligned_length_samples": int(len(ref)),
        "aligned_length_sec": float(len(ref) / args.sr),
        "metrics": metrics,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.report:
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
