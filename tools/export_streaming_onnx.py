import argparse
import json
import logging
import os
import sys

now_dir = os.getcwd()
sys.path.append(now_dir)

from infer.modules.onnx.export_streaming import export_streaming_onnx


logging.basicConfig(level=logging.INFO)


def main():
    parser = argparse.ArgumentParser(
        description="Export a fixed-shape streaming RVC ONNX model for C++/ORT inference."
    )
    parser.add_argument("--model", required=True, help="Path to .pth model")
    parser.add_argument("--output", required=True, help="Path to output .onnx")
    parser.add_argument(
        "--block-time",
        type=float,
        default=0.25,
        help="Realtime block_time in seconds",
    )
    parser.add_argument(
        "--crossfade-time",
        type=float,
        default=0.05,
        help="Realtime crossfade_time in seconds",
    )
    parser.add_argument(
        "--extra-time",
        type=float,
        default=2.5,
        help="Realtime extra_time in seconds",
    )
    parser.add_argument(
        "--formant-shift",
        type=float,
        default=0.0,
        help="Realtime formant shift in semitones",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip onnx.checker and onnxruntime validation",
    )
    args = parser.parse_args()

    result = export_streaming_onnx(
        model_path=args.model,
        output_path=args.output,
        block_time=args.block_time,
        crossfade_time=args.crossfade_time,
        extra_time=args.extra_time,
        formant_shift=args.formant_shift,
        verify=not args.no_verify,
    )
    logging.info(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
