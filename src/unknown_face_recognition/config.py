"""Project paths and model metadata."""

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_KNOWN_DIR = PROJECT_ROOT / "data" / "known_faces"
DEFAULT_GALLERY = PROJECT_ROOT / "data" / "gallery.npz"


@dataclass(frozen=True)
class ModelSpec:
    filename: str
    url: str
    minimum_bytes: int


MODEL_SPECS = {
    "yunet": ModelSpec(
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        minimum_bytes=100_000,
    ),
    "sface": ModelSpec(
        filename="face_recognition_sface_2021dec.onnx",
        url=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        minimum_bytes=1_000_000,
    ),
}

SUPPORTED_IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_COSINE_THRESHOLD = 0.363
