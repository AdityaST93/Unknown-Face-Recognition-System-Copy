"""Evaluate a built gallery on consented, held-out images by identity folder."""

import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from .config import SUPPORTED_IMAGE_SUFFIXES
from .errors import ProjectError
from .face_engine import FaceEngine
from .gallery import Gallery


def evaluate_gallery(gallery: Gallery, engine: FaceEngine, evaluation_dir: Path,
                     threshold: float) -> dict[str, Any]:
    """Compute open-set evaluation counts and rates for a held-out image folder."""
    if not evaluation_dir.is_dir():
        raise ProjectError(f"Evaluation folder does not exist: {evaluation_dir}")
    known_names = {name.casefold() for name in gallery.names}
    counts: Counter[str] = Counter()
    skipped = 0

    for identity_dir in sorted(path for path in evaluation_dir.iterdir() if path.is_dir()):
        identity = identity_dir.name.replace("_", " ").casefold()
        is_unknown = identity in {"unknown", "unregistered"} or identity not in known_names
        for image_path in sorted(
            path for path in identity_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        ):
            image = cv2.imread(str(image_path))
            faces = engine.detect(image) if image is not None else []
            if len(faces) != 1:
                skipped += 1
                continue
            match = gallery.match(faces[0].embedding, threshold)
            predicted_name = match.name.casefold() if match.name else None
            counts["evaluated"] += 1

            if is_unknown:
                if predicted_name is None:
                    counts["unknown_correctly_rejected"] += 1
                else:
                    counts["unknown_false_accepts"] += 1
            elif predicted_name == identity:
                counts["known_correct"] += 1
            elif predicted_name is None:
                counts["known_false_rejects"] += 1
            else:
                counts["known_wrong_identity"] += 1

    known_total = sum(counts[key] for key in (
        "known_correct", "known_false_rejects", "known_wrong_identity"
    ))
    unknown_total = counts["unknown_correctly_rejected"] + counts["unknown_false_accepts"]
    evaluated = counts["evaluated"]
    correct = counts["known_correct"] + counts["unknown_correctly_rejected"]
    report: dict[str, Any] = {
        "threshold": threshold,
        "evaluated_images": evaluated,
        "skipped_images": skipped,
        "known_images": known_total,
        "unknown_images": unknown_total,
        "known_correct": counts["known_correct"],
        "known_false_rejects": counts["known_false_rejects"],
        "known_wrong_identity": counts["known_wrong_identity"],
        "unknown_correctly_rejected": counts["unknown_correctly_rejected"],
        "unknown_false_accepts": counts["unknown_false_accepts"],
        "accuracy": correct / evaluated if evaluated else None,
        "known_false_reject_rate": (
            counts["known_false_rejects"] / known_total if known_total else None
        ),
        "known_wrong_identity_rate": (
            counts["known_wrong_identity"] / known_total if known_total else None
        ),
        "unknown_false_accept_rate": (
            counts["unknown_false_accepts"] / unknown_total if unknown_total else None
        ),
    }
    if not evaluated:
        raise ProjectError(
            f"No evaluable images found in {evaluation_dir}. Use one detectable face per image."
        )
    return report


def save_report(report: dict[str, Any], output: Path) -> None:
    """Write a machine-readable UTF-8 JSON evaluation report."""
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".part")
    temporary.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
