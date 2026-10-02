"""Sketch-to-CAD. Importing the package caps image decoding for the whole process (audit H1): OpenCV reads
OPENCV_IO_MAX_IMAGE_PIXELS on its first decode, so a PNG of a few kilobytes that unpacks to gigabytes is refused."""
import os

__version__ = "0.1.0"
MAX_PIXELS = int(os.environ.get("S2C_MAX_PIXELS", "40000000"))
os.environ.setdefault("OPENCV_IO_MAX_IMAGE_PIXELS", str(MAX_PIXELS))
