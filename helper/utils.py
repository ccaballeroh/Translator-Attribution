"""Util functions to clean directory during processing and testing."""

from pathlib import Path


def cleaning(*, path: Path, extension: str, affix: str) -> None:
    """Deletes all files in a Path with certain extension and name termination.
    
    Inputs:
    path:       Path    - directory where to delete files
    extension:  str     - extension of files to errase
    affix:      str     - temination of file name to delete

    Returns:
    None
    """
    assert path.is_dir(), f"Path: {path} does not exist"
    for filename in path.iterdir():
        if filename.suffix == extension and filename.stem.endswith(affix):
            filename.unlink()
    return None


def clean_example():
    """Deletes all files created during processing or corpora no longer needed during experiments."""
    JSON_FOLDER = Path(r"./auxfiles/json/")
    cleaning(path=JSON_FOLDER, extension=".json", affix="trash")

    CORPORA = Path(r"./Corpora/Proc_Quixote")
    if CORPORA.exists():
        cleaning(path=CORPORA, extension=".txt", affix="proc")
        CORPORA.rmdir()

    CORPORA = Path(r"./Corpora/Proc_Ibsen")
    if CORPORA.exists():
        cleaning(path=CORPORA, extension=".txt", affix="proc")
        CORPORA.rmdir()

    SN_FOLDER = Path(r"./auxfiles/txt")
    cleaning(path=(SN_FOLDER / "Quixote"), extension=".txt", affix="sn")
    cleaning(path=(SN_FOLDER / "Ibsen"), extension=".txt", affix="sn")


if __name__ == "__main__":
    clean_example()
