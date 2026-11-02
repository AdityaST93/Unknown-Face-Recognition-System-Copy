"""Command line interface for downloading, enrolling, indexing, and recognizing."""

import argparse
from pathlib import Path

from .config import (
    DEFAULT_COSINE_THRESHOLD,
    DEFAULT_GALLERY,
    DEFAULT_KNOWN_DIR,
    DEFAULT_MODEL_DIR,
)
from .enrollment import enroll_from_camera
from .errors import ProjectError
from .evaluation import evaluate_gallery, save_report
from .face_engine import FaceEngine
from .gallery import Gallery, build_gallery
from .model_manager import download_models, resolve_models
from .runtime import run_recognition


def _add_model_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR,
                        help=f"ONNX model directory (default: {DEFAULT_MODEL_DIR})")


def _engine(model_dir: Path, detection_threshold: float) -> FaceEngine:
    yunet, sface = resolve_models(model_dir)
    return FaceEngine(yunet, sface, detection_threshold)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="unknown-face",
        description="Enroll known people and recognize them locally; unmatched faces are Unknown.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    models = commands.add_parser("download-models", help="Download YuNet and SFace ONNX models")
    _add_model_options(models)

    enroll = commands.add_parser("enroll", help="Capture face samples for a known person")
    enroll.add_argument("--name", required=True, help="Display name to enroll")
    enroll.add_argument("--source", default="0", help="Camera index or video path (default: 0)")
    enroll.add_argument("--known-dir", type=Path, default=DEFAULT_KNOWN_DIR)
    enroll.add_argument("--detection-threshold", type=float, default=0.8)
    _add_model_options(enroll)

    build = commands.add_parser(
        "build-gallery", help="Build the embedding gallery from known-face folders"
    )
    build.add_argument("--known-dir", type=Path, default=DEFAULT_KNOWN_DIR)
    build.add_argument("--output", type=Path, default=DEFAULT_GALLERY)
    build.add_argument("--detection-threshold", type=float, default=0.8)
    _add_model_options(build)

    run = commands.add_parser("run", help="Recognize known people and flag unknown faces")
    run.add_argument("--source", default="0", help="Camera index or video path (default: 0)")
    run.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    run.add_argument("--threshold", type=float, default=DEFAULT_COSINE_THRESHOLD,
                     help="Minimum cosine similarity for a known match (default: 0.363)")
    run.add_argument("--detection-threshold", type=float, default=0.8)
    run.add_argument("--log-unknown", type=Path, default=None,
                     help=("Optional CSV path for unknown timestamps and scores; "
                           "no images are saved"))
    _add_model_options(run)

    evaluate = commands.add_parser("evaluate", help="Measure open-set errors on held-out images")
    evaluate.add_argument("--evaluation-dir", type=Path, required=True,
                          help=("Held-out images in one folder per identity; use Unknown "
                                "for unregistered people"))
    evaluate.add_argument("--gallery", type=Path, default=DEFAULT_GALLERY)
    evaluate.add_argument("--threshold", type=float, default=DEFAULT_COSINE_THRESHOLD)
    evaluate.add_argument(
        "--report", type=Path, default=None, help="Optional JSON report output path"
    )
    evaluate.add_argument("--detection-threshold", type=float, default=0.8)
    _add_model_options(evaluate)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "download-models":
            for name, path in download_models(args.model_dir).items():
                print(f"{name}: {path}")
        elif args.command == "enroll":
            engine = _engine(args.model_dir, args.detection_threshold)
            count = enroll_from_camera(args.name, args.known_dir, args.source, engine)
            print(f"Enrollment photos saved. Total samples for {args.name}: {count}")
            print("Run `unknown-face build-gallery` to refresh the searchable gallery.")
        elif args.command == "build-gallery":
            engine = _engine(args.model_dir, args.detection_threshold)
            gallery, skipped = build_gallery(args.known_dir, args.output, engine)
            print(f"Saved {len(gallery.names)} face embeddings for "
                  f"{len(set(gallery.names))} people to {args.output}")
            if skipped:
                print(f"Skipped {skipped} image(s); review the messages above.")
        elif args.command == "run":
            if not 0.0 <= args.threshold <= 1.0:
                parser.error("--threshold must be between 0 and 1")
            engine = _engine(args.model_dir, args.detection_threshold)
            gallery = Gallery.load(args.gallery)
            run_recognition(args.source, gallery, engine, args.threshold, args.log_unknown)
        elif args.command == "evaluate":
            if not 0.0 <= args.threshold <= 1.0:
                parser.error("--threshold must be between 0 and 1")
            engine = _engine(args.model_dir, args.detection_threshold)
            gallery = Gallery.load(args.gallery)
            report = evaluate_gallery(gallery, engine, args.evaluation_dir, args.threshold)
            for key, value in report.items():
                print(f"{key}: {value}")
            if args.report:
                save_report(report, args.report)
                print(f"JSON report saved to {args.report}")
    except ProjectError as exc:
        parser.exit(2, f"error: {exc}\n")
