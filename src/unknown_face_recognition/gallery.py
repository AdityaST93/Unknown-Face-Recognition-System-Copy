"""Build, persist, and query an enrolled identity gallery."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import SUPPORTED_IMAGE_SUFFIXES
from .errors import ProjectError
from .face_engine import FaceEngine


@dataclass(frozen=True)
class Match:
    name: str | None
    score: float


@dataclass
class Gallery:
    names: list[str]
    embeddings: np.ndarray

    @classmethod
    def load(cls, path: Path) -> "Gallery":
        if not path.is_file():
            raise ProjectError(
                f"Gallery not found: {path}. Run `unknown-face build-gallery` first."
            )
        try:
            with np.load(path, allow_pickle=False) as data:
                names = data["names"].astype(str).tolist()
                embeddings = data["embeddings"].astype(np.float32)
        except (OSError, ValueError, KeyError) as exc:
            raise ProjectError(f"Could not read gallery {path}: {exc}") from exc
        if embeddings.ndim != 2 or embeddings.shape[0] != len(names) or not names:
            raise ProjectError(f"Gallery is empty or invalid: {path}")
        return cls(names=names, embeddings=embeddings)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".part")
        with temporary.open("wb") as output:
            np.savez_compressed(
                output,
                names=np.asarray(self.names, dtype=np.str_),
                embeddings=np.asarray(self.embeddings, dtype=np.float32),
                format_version=np.asarray([1], dtype=np.int32),
            )
        temporary.replace(path)

    def match(self, embedding: np.ndarray, threshold: float) -> Match:
        """Return the most similar enrolled sample if it clears the cutoff."""
        scores = self.embeddings @ embedding.astype(np.float32)
        best_index = int(np.argmax(scores))
        best_score = float(scores[best_index])
        return Match(self.names[best_index] if best_score >= threshold else None, best_score)


def build_gallery(known_dir: Path, output: Path, engine: FaceEngine) -> tuple[Gallery, int]:
    """Embed every usable image in one folder per identity and persist a gallery."""
    if not known_dir.is_dir():
        raise ProjectError(f"Known-face folder does not exist: {known_dir}")
    embeddings: list[np.ndarray] = []
    names: list[str] = []
    skipped = 0
    for person_dir in sorted(path for path in known_dir.iterdir() if path.is_dir()):
        image_paths = sorted(
            path for path in person_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
        )
        if not image_paths:
            print(f"Skipping empty identity folder: {person_dir.name}")
        for image_path in image_paths:
            image = cv2.imread(str(image_path))
            faces = engine.detect(image) if image is not None else []
            if len(faces) != 1:
                print(f"Skipping {image_path}: expected one detectable face, found {len(faces)}")
                skipped += 1
                continue
            embeddings.append(faces[0].embedding)
            names.append(person_dir.name.replace("_", " "))

    if not embeddings:
        raise ProjectError(
            f"No usable face images found under {known_dir}. Add folders by person, "
            "with one clear face per image."
        )
    gallery = Gallery(names=names, embeddings=np.vstack(embeddings).astype(np.float32))
    gallery.save(output)
    return gallery, skipped
