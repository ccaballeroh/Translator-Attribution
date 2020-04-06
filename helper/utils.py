from pathlib import Path


def cleaning(*, path: Path, extension: str, affix: str) -> None:
    assert path.is_dir(), f"Path: {path} does not exist"
    for filename in path.iterdir():
        if filename.suffix == extension and filename.stem.endswith(affix):
            filename.unlink()
    return None


def clean_example():
    JSON_FOLDER = Path(r"./auxfiles/json/")
    cleaning(path=JSON_FOLDER, extension=".json", affix="trash")

    CORPORA = Path(r"./Corpora/Proc_Quixote")
    cleaning(path=CORPORA, extension=".txt", affix="proc")

    CORPORA = Path(r"./Corpora/Proc_Ibsen")
    cleaning(path=CORPORA, extension=".txt", affix="proc")

    SN_FOLDER = Path(r"./auxfiles/txt")
    cleaning(path=(SN_FOLDER / "Quixote"), extension=".txt", affix="sn")
    cleaning(path=(SN_FOLDER / "Ibsen"), extension=".txt", affix="sn")


if __name__ == "__main__":
    clean_example()
