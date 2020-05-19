__all__ = ["analysis", "preprocessing", "features", "utils", "IN_COLAB", "ROOT"]
__version__ = "1.0"
__author__ = "Christian Caballero <cdch10@gmail.com>"

from pathlib import Path
import sys

# flag to change root folder if running in colab
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB:
    ROOT = Path(r"./drive/My Drive/Translator-Attribution/")  # Root folder in colab
    print("In colab!")
else:
    ROOT = Path(r".")  # Root folder if running locally
