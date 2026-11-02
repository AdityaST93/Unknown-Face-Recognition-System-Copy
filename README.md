# Unknown Face Recognition System

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-YuNet%20%2B%20SFace-5C3EE8?logo=opencv&logoColor=white)
![Runs locally](https://img.shields.io/badge/processing-local-16825D)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

A local, open-set face recognition demo. Enroll people you know, then run it on a webcam or video. The system labels a face with the closest enrolled name only when its similarity passes a configurable threshold; otherwise it displays **Unknown**.

Face detection and recognition use the OpenCV Zoo **YuNet** and **SFace** ONNX models. Frames are processed on the computer running the app. By default, the project saves no video, screenshots, or recognition history.

> **Responsible use:** Face recognition processes biometric data. Enroll people only with their knowledge and consent. This educational demo can make mistakes and is not an identity verification, access control, or surveillance product.

## Features

- Open-set matching: unmatched faces are explicitly reported as **Unknown**.
- Webcam enrollment and recognition, with video-file input supported.
- Multiple people and multiple enrollment images per person.
- Local embedding gallery stored as a compact NumPy archive.
- Adjustable cosine-similarity and face-detection thresholds.
- Optional CSV event log with UTC timestamps and scores; it never saves face images.
- Downloadable model files are kept outside Git and checked for plausible size.
- Modular source layout, typed data objects, CLI help, and clear input errors.

## Quick start (Windows PowerShell)

```powershell
cd Unknown-Face-Recognition-System
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
unknown-face download-models
```

If the `unknown-face` command is not found in the current shell, use the module form:

```powershell
python -m unknown_face_recognition download-models
```

The first model download needs an internet connection. After the models are in `models/`, detection and recognition run locally. See [installation notes](docs/installation.md) for macOS/Linux and troubleshooting.

## Run from VS Code

1. In VS Code, choose **File → Open Folder** and open this entire `Unknown-Face-Recognition-System` folder.
2. Install the recommended Microsoft Python and Pylance extensions if VS Code offers them.
3. Choose **Terminal → Run Task... → Install project in virtual environment**. The task creates `.venv` and installs this project with its dependencies.
4. If needed, choose **Python: Select Interpreter** from the Command Palette and select `.venv`.
5. Open **Run and Debug** (`Ctrl+Shift+D`) and select **1. Download YuNet + SFace models**. This first step downloads the model files once.
6. Run **2. Enroll person (edit name)** and change `Aarav` in `.vscode/launch.json` to the person's name. Press **S** in the camera window to save face samples, then **Q** to finish.
7. Run **3. Build face gallery**, then **4. Run webcam recognition**.

The camera preview is its own OpenCV window; keep it open for recognition and press **Q** to exit. VS Code shows each command's output in its Debug Console.

## Add known people

Create one folder per person under `data/known_faces/`, using a clear display name. Enroll from a webcam:

```powershell
unknown-face enroll --name "Aarav"
unknown-face enroll --name "Meera"
```

Face the camera and press **S** to save a sample; change your angle or lighting between samples. Press **Q** to finish. About 10-20 varied, clear samples per person are a useful starting point. The software will not decide who a person is from a single “best” sample: it builds a local embedding gallery from the enrolled photos.

You can also place images into folders manually:

```text
data/known_faces/
├── Aarav/
│   ├── front.jpg
│   └── side.jpg
└── Meera/
    ├── front.jpg
    └── daylight.jpg
```

Use one visible face per image. Use clear, consented images; avoid group photos and very small, blurred, or heavily occluded faces. Images remain on your machine.

## Build and run

After adding or changing enrollment photos, rebuild the gallery:

```powershell
unknown-face build-gallery
```

Start the webcam recognizer and press **Q** in the preview window to exit:

```powershell
unknown-face run
```

Other useful commands:

```powershell
# Use a selected webcam index or a video file
unknown-face run --source 1
unknown-face run --source .\sample_video.mp4

# Make recognition stricter or more permissive
unknown-face run --threshold 0.42

# Optional, rate-limited CSV records; no images are saved
unknown-face run --log-unknown .\logs\unknown_events.csv

# Evaluate on held-out images (no evaluation images are committed)
unknown-face evaluate --evaluation-dir D:\vision-data\held-out --report .\output\evaluation.json

# Keep models, photos, or gallery outside the repository
unknown-face download-models --model-dir D:\vision-models
unknown-face build-gallery --known-dir D:\my-consented-faces --output D:\vision-data\gallery.npz
unknown-face run --model-dir D:\vision-models --gallery D:\vision-data\gallery.npz

# See all options
unknown-face --help
unknown-face run --help
```

## How it works

1. **Detect:** YuNet locates faces and returns bounding boxes plus five landmarks.
2. **Align and embed:** SFace aligns each face using the landmarks and converts it to a feature vector.
3. **Compare:** the feature is compared with each enrolled sample using cosine similarity.
4. **Decide:** the highest score at or above the threshold is labeled with that sample's person; lower scores are **Unknown**.

The default `0.363` cutoff is OpenCV's published SFace operating point for its LFW evaluation example, not a universal guarantee. Camera, lighting, population, and sample quality change the error trade-off. Read [threshold tuning](docs/threshold-tuning.md) before interpreting scores. A similarity score is not a probability.

## Project layout

```text
Unknown-Face-Recognition-System/
├── src/unknown_face_recognition/  # CLI, models, gallery, enrollment, runtime
├── data/known_faces/              # Local consented enrollment photos (ignored by Git)
├── models/                        # Downloaded ONNX files (ignored by Git)
├── docs/                          # Architecture, installation, privacy, thresholds
├── .github/                       # Issue template and lint workflow
├── pyproject.toml                 # Package metadata and CLI entry point
└── README.md
```

Read [architecture](docs/architecture.md), [installation](docs/installation.md), [privacy](docs/privacy.md), [threshold tuning](docs/threshold-tuning.md), and [model licenses](MODEL_LICENSES.md).
The GitHub Ruff workflow runs on pushes and pull requests.

## Limitations

- Recognition can falsely match or reject a person. Do not use the output to make consequential decisions.
- The default threshold has not been calibrated for your webcam or enrolled population.
- Extreme pose, masks, low resolution, backlighting, motion blur, and face occlusion can reduce detection or matching quality.
- The demo has no liveness detection. A photo or screen may be accepted as a face.
- There is no multi-object identity tracker; labels are computed independently on each frame.
- A CSV event log is optional, records at most one unknown event per second, and can still contain sensitive data.

## Credits

This project uses OpenCV's YuNet face detector and SFace face recognizer. Model creators and their licenses are documented in [MODEL_LICENSES.md](MODEL_LICENSES.md). The OpenCV Zoo examples and model cards are linked there.
