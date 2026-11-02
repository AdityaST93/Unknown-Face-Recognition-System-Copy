# Privacy and data handling

This demo performs face processing locally. It does not upload frames, names, face embeddings, or event logs to a service.

## Files created by the project

| Path | Contents | Default Git status |
|---|---|---|
| `data/known_faces/<name>/` | Enrollment face crops | Ignored |
| `data/gallery.npz` | Face embeddings and identity labels | Ignored |
| `models/*.onnx` | YuNet and SFace model weights | Ignored |
| Optional `--log-unknown` CSV | UTC time, similarity score, camera source | Ignored under `logs/` |

Do not commit real people's enrollment images, embedding galleries, or event logs. Deleting a person means deleting their source folder and rebuilding the gallery. Securely delete copies and backups according to your own data retention requirements.

Use only images captured or collected with permission. Limit access to the project directory, avoid public CCTV or sensitive footage, and obtain appropriate legal/privacy review before any real-world deployment. This prototype does not provide consent management, access controls, encryption-at-rest, liveness checks, or retention automation.
