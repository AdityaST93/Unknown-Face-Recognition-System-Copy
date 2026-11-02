# Threshold tuning

The recognizer returns cosine similarity. Higher values mean more similar feature vectors, but the score is not a calibrated probability.

The default threshold is `0.363`, which is the operating point used in OpenCV's SFace example for its LFW evaluation. It is a starting value only. Do not assume that it has the same error rates for a different camera, age group, image quality, or environment.

## Suggested evaluation process

1. Enroll people with images that are separate from the images used for evaluation.
2. Gather consented validation images from the expected environment and include people who are not enrolled.
3. For every validation face, record whether it is the same enrolled person and its highest gallery similarity.
4. Sweep thresholds across the observed score range. Count false accepts (unknown person assigned a known name) and false rejects (known person labeled Unknown).
5. Select and document the operating point that fits the demo's intended, low-risk use. Re-evaluate when camera, lighting, or enrollment data changes.

Lowering `--threshold` reduces rejections but tends to increase false accepts. Raising it tends to reject more genuine known faces. Use a distinct validation set; tuning on enrollment images alone gives an overly optimistic view.

Use `unknown-face evaluate --evaluation-dir PATH` to measure known false rejects, known wrong-identity assignments, and unknown false accepts on held-out images. The folder names must match gallery identity names; `Unknown` and `Unregistered` folders, or names not in the gallery, are treated as unregistered people. Keep evaluation photos separate from enrollment photos. Review [OpenCV's DNN face tutorial](https://docs.opencv.org/4.x/d0/dd4/tutorial_dnn_face.html) for the model's published example thresholds and evaluation context.
