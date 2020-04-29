__all__ = ["analysis", "preprocessing", "utils", "IN_COLAB", "ROOT"]
__version__ = "1.0"
__author__ = "Christian Caballero <cdch10@gmail.com>"

from pathlib import Path
import sys

IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    ROOT = Path(r"./drive/My Drive/00Tesis/")
    print("In colab!")
else:
    ROOT = Path(r".")
