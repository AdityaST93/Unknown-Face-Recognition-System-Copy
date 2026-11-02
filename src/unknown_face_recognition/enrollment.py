"""Interactive webcam enrollment into a folder-based identity gallery."""

import re
from pathlib import Path

import cv2

from .errors import ProjectError
from .face_engine import FaceEngine


def enroll_from_camera(name: str, known_dir: Path, source: str, engine: FaceEngine) -> int:
    """Capture face crops for one person. Press S to save, Q to finish."""
    display_name = " ".join(name.split())
    folder_name = re.sub(r"[^\w-]+", "_", display_name, flags=re.UNICODE).strip("_")
    if not display_name or not folder_name:
        raise ProjectError("Enter a name containing letters or numbers.")
    person_dir = known_dir / folder_name
    person_dir.mkdir(parents=True, exist_ok=True)
    video_source = int(source) if source.isdigit() else source
    capture = cv2.VideoCapture(video_source)
    if not capture.isOpened():
        raise ProjectError(f"Could not open camera/video source: {source}")

    saved = len(list(person_dir.glob("*.jpg")))
    print("Press S to save the largest detected face; Q to finish enrollment.")
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            faces = engine.detect(frame)
            for face in faces:
                x, y, width, height = face.box
                cv2.rectangle(frame, (x, y), (x + width, y + height), (40, 210, 60), 2)
            cv2.putText(frame, f"{display_name} | saved: {saved} | S save / Q quit",
                        (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 210, 60), 2)
            cv2.imshow("Enroll known person", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if len(faces) != 1:
                    print("Show exactly one face before saving a sample.")
                    continue
                face = faces[0]
                x, y, width, height = face.box
                height_frame, width_frame = frame.shape[:2]
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(width_frame, x + width), min(height_frame, y + height)
                crop = frame[y1:y2, x1:x2]
                if crop.size == 0:
                    print("Face crop was empty; try again.")
                    continue
                saved += 1
                path = person_dir / f"sample_{saved:04d}.jpg"
                if not cv2.imwrite(str(path), crop):
                    saved -= 1
                    print(f"Could not save image: {path}")
                else:
                    print(f"Saved {path.name}")
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return saved
