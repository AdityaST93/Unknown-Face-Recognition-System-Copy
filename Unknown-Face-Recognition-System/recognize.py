"""Compatibility launcher: use `unknown-face run` for options."""

import sys

from unknown_face_recognition.cli import main

if __name__ == "__main__":
    main(["run", *sys.argv[1:]])
