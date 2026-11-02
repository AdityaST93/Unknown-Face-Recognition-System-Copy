# Architecture

## Data flow

```text
camera/video frame
       │
       ▼
YuNet detector ── face box + 5 landmarks
       │
       ▼
SFace alignCrop + feature ── normalized face embedding
       │
       ├── enrolled sample vectors in gallery.npz
       ▼
cosine similarity search ── best score ≥ threshold ? name : Unknown
```

## Components

- `cli.py` parses commands and converts expected user errors into concise terminal messages.
- `model_manager.py` fetches the named ONNX files from the OpenCV Zoo media host and writes them atomically after a minimum-size check.
- `face_engine.py` owns the OpenCV YuNet detector and SFace recognizer; its output is a `DetectedFace` containing a box, detector confidence, and normalized feature vector.
- `enrollment.py` captures a face crop to the person's local data folder.
- `gallery.py` converts folders of images into a portable NPZ archive. Each valid photo contributes one labeled embedding; gallery matching searches all samples for the highest cosine similarity.
- `runtime.py` owns the video capture loop, preview, and optional unknown-event CSV.
- `config.py` centralizes paths, supported image types, model filenames, and defaults.

## Gallery format

The gallery is a compressed NumPy archive containing Unicode identity names, a float32 embedding matrix, and a format version. It contains biometric templates, so it is excluded from Git and should be treated as sensitive. It can be rebuilt from the consented source images.

## Design decisions

- YuNet and SFace are downloaded on request. Their binary weights are not committed to the repository.
- Identity data stays local. No web API, database server, or telemetry is used.
- Unknown event logging is opt-in, rate limited, and stores no frame or embedding.
- A nearest enrolled sample is used rather than one centroid per person so changes in pose and lighting can have multiple references. This increases matching work linearly with enrollment sample count.
