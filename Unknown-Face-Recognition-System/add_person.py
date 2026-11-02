"""Compatibility launcher: use `unknown-face enroll --name NAME` for options."""

import sys

from unknown_face_recognition.cli import main

if __name__ == "__main__":
    main(["enroll", *sys.argv[1:]])
