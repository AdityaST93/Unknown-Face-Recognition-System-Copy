"""Live camera/video recognition loop and optional unknown-event log."""

import csv
import time
from datetime import datetime, timezone
from pathlib import Path

import cv2

from .errors import ProjectError
from .face_engine import FaceEngine
from .gallery import Gallery


def run_recognition(source: str, gallery: Gallery, engine: FaceEngine,
                    threshold: float, event_log: Path | None = None) -> None:
    """Display recognition results. Press Q to stop; frames stay in memory."""
    video_source = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        raise ProjectError(f"Could not open camera/video source: {source}")

    log_writer = None
    log_file = None
    if event_log:
        event_log.parent.mkdir(parents=True, exist_ok=True)
        needs_header = not event_log.exists() or event_log.stat().st_size == 0
        log_file = event_log.open("a", newline="", encoding="utf-8")
        log_writer = csv.writer(log_file)
        if needs_header:
            log_writer.writerow(["timestamp_utc", "cosine_similarity", "camera_source"])

    print("Recognition is running. Press Q in the preview window to quit.")
    last_unknown_log = 0.0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            for face in engine.detect(frame):
                match = gallery.match(face.embedding, threshold)
                x, y, width, height = face.box
                is_known = match.name is not None
                label = match.name if is_known else "Unknown"
                color = (40, 210, 60) if is_known else (40, 50, 230)
                cv2.rectangle(frame, (x, y), (x + width, y + height), color, 2)
                cv2.putText(frame, f"{label} | cosine {match.score:.3f}",
                            (x, max(22, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.58, color, 2)
                now = time.monotonic()
                if log_writer and not is_known and now - last_unknown_log >= 1.0:
                    log_writer.writerow([
                        datetime.now(timezone.utc).isoformat(timespec="seconds"),
                        f"{match.score:.5f}", source,
                    ])
                    last_unknown_log = now
            cv2.imshow("Unknown Face Recognition System", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
        if log_file:
            log_file.close()
