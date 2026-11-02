# Installation and troubleshooting

## Requirements

- Python 3.10 or newer
- A webcam for live capture, or a supported image/video path where stated
- Internet only for initial model downloads and package installation
- OpenCV 4.10+ with `FaceDetectorYN` and `FaceRecognizerSF`

## Windows PowerShell

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
unknown-face download-models
```

If PowerShell blocks virtual-environment activation, run the executable directly instead:

```powershell
.\.venv\Scripts\python.exe -m pip install -e .
.\.venv\Scripts\python.exe -m unknown_face_recognition download-models
```

## macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
unknown-face download-models
```

## Troubleshooting

### `FaceDetectorYN` or `FaceRecognizerSF` is missing

Install this project into a clean environment. Avoid installing both `opencv-python` and `opencv-contrib-python` into the same environment: both provide the `cv2` module.

### Camera does not open

Close other camera apps and try another camera index, for example `unknown-face run --source 1`. Confirm the camera is available to Python and that the operating system has granted permission.

### No faces are detected

Improve lighting, face the camera, move closer, reduce motion blur, or lower the detector threshold slightly with `--detection-threshold 0.7`. This controls detection confidence and is separate from recognition similarity.

### A known person shows as Unknown

Add clear samples in varied lighting and angles, rebuild the gallery, and tune the cosine threshold using separate validation photos. Lower values accept more matches and may increase false matches. See [threshold tuning](threshold-tuning.md).

### ONNX model download fails

Retry with an internet connection and confirm the model files in `models/` are not zero-byte or partial files. Remove a partial/corrupt `.onnx` file and run `unknown-face download-models` again.
