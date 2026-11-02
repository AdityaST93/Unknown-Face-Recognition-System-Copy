# Contributing

Thanks for helping improve the project. Keep changes focused and explain the user-visible behavior they affect.

## Before opening a pull request

- Describe the problem and the proposed change.
- Keep real faces, galleries, event logs, model weights, and other private data out of commits.
- Update the README or relevant docs when commands or behavior change.
- Include the Python version, OpenCV version, operating system, and reproduction steps for bug reports.
- For changes to matching or detection, describe the data and threshold assumptions; do not claim accuracy without a documented evaluation.

## Local development

```bash
python -m venv .venv
# Activate the environment, then:
python -m pip install -e .
python -m pip install ruff
ruff check src
```

Use consented sample data only. Do not upload enrollment images in issues or pull requests.
