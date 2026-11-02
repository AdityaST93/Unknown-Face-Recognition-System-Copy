"""YuNet face detection and SFace embedding extraction."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .errors import ProjectError


@dataclass(frozen=True)
class DetectedFace:
    """A detected face row plus its aligned SFace embedding."""

    box: tuple[int, int, int, int]
    confidence: float
    embedding: np.ndarray


class FaceEngine:
    """Wrap the OpenCV YuNet detector and SFace feature extractor."""

    def __init__(self, detector_model: Path, recognizer_model: Path,
                 detection_threshold: float = 0.8) -> None:
        if not hasattr(cv2, "FaceDetectorYN") or not hasattr(cv2, "FaceRecognizerSF"):
            raise ProjectError(
                "This OpenCV build does not include FaceDetectorYN/FaceRecognizerSF. "
                "Install the project dependencies in a clean virtual environment."
            )
        try:
            self.detector = cv2.FaceDetectorYN.create(
                str(detector_model), "", (320, 320), detection_threshold, 0.3, 5000
            )
            self.recognizer = cv2.FaceRecognizerSF.create(str(recognizer_model), "")
        except cv2.error as exc:
            raise ProjectError(f"Could not load ONNX models: {exc}") from exc

    def detect(self, frame: np.ndarray) -> list[DetectedFace]:
        """Detect and embed every face found in one BGR frame."""
        if frame is None or frame.size == 0:
            return []
        height, width = frame.shape[:2]
        self.detector.setInputSize((width, height))
        _, rows = self.detector.detect(frame)
        if rows is None:
            return []

        result = []
        for row in rows:
            try:
                aligned = self.recognizer.alignCrop(frame, row)
                feature = self.recognizer.feature(aligned).astype(np.float32).reshape(-1)
            except cv2.error:
                continue
            norm = float(np.linalg.norm(feature))
            if norm == 0:
                continue
            x, y, box_width, box_height = (int(round(v)) for v in row[:4])
            result.append(DetectedFace(
                box=(x, y, box_width, box_height),
                confidence=float(row[-1]),
                embedding=feature / norm,
            ))
        return result
