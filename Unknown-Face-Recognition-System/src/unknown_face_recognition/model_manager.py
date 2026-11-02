"""Download and verify the small set of required OpenCV Zoo model files."""

from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from .config import MODEL_SPECS, ModelSpec
from .errors import ProjectError


def _download_one(spec: ModelSpec, destination: Path, timeout: int = 60) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size >= spec.minimum_bytes:
        return destination

    temporary = destination.with_suffix(destination.suffix + ".part")
    request = Request(spec.url, headers={"User-Agent": "UnknownFaceRecognition/0.2"})
    try:
        with urlopen(request, timeout=timeout) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
    except (OSError, URLError) as exc:
        temporary.unlink(missing_ok=True)
        raise ProjectError(f"Could not download {spec.filename}: {exc}") from exc

    if temporary.stat().st_size < spec.minimum_bytes:
        size = temporary.stat().st_size
        temporary.unlink(missing_ok=True)
        raise ProjectError(
            f"Downloaded {spec.filename} is only {size} bytes; expected a model file. "
            "Check your connection and try again."
        )
    temporary.replace(destination)
    return destination


def download_models(model_dir: Path) -> dict[str, Path]:
    """Download both required models atomically and return their local paths."""
    return {
        name: _download_one(spec, model_dir / spec.filename)
        for name, spec in MODEL_SPECS.items()
    }


def resolve_models(model_dir: Path) -> tuple[Path, Path]:
    """Return model paths or explain how to install the missing model files."""
    yunet = model_dir / MODEL_SPECS["yunet"].filename
    sface = model_dir / MODEL_SPECS["sface"].filename
    missing = [path.name for path in (yunet, sface) if not path.is_file()]
    if missing:
        raise ProjectError(
            "Required model file(s) are missing: " + ", ".join(missing) +
            ". Run `unknown-face download-models` first."
        )
    return yunet, sface
