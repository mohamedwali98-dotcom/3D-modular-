"""Sketch-to-CAD. Importing the package caps image decoding for the whole process (audit H1): a PNG of a few
kilobytes that unpacks to gigabytes is refused. OpenCV reads OPENCV_IO_MAX_IMAGE_PIXELS when cv2 is imported, so
s2c must be imported first; every entry point does, and a warning says when it was not. S2C_MAX_PIXELS is the one
setting, from the environment or the .env file (read here: the app loads .env only after its modules)."""
import os
import sys
import warnings

from s2c.config import setting as _setting

__version__ = "0.1.0"
MAX_PIXELS = int(_setting("S2C_MAX_PIXELS", "40000000"))
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(MAX_PIXELS)
if "cv2" in sys.modules:
    warnings.warn("cv2 was imported before s2c, so OPENCV_IO_MAX_IMAGE_PIXELS may not cap image decoding: "
                  "import s2c first", stacklevel=2)
