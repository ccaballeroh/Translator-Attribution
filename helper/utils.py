"""Util functions"""

from collections import defaultdict
from helper import ROOT
from pathlib import Path
from pandas import DataFrame
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from typing import Dict, List


def return_n_most_important(
    *, clf: LogisticRegression, v: DictVectorizer, encoder: LabelEncoder, n: int = 10
) -> Dict[str, DataFrame]:
    """Returns n features with largest weights in a logistic regression classifier.

    As inputs a trained logistic regressor along with the DictVectorizer and
    LabelEncoder used to encode dictionary of feature names and values and
    classes names, and the number of features to return.

    Parameters:
    clf:        LogisticRegression  - A trained logistic regression classifier
    v:          DictVectorizer      - Scikit-learn DictVectorizer used for transforming
                                    dict of features and counts to a numpy array
    encoder:    LabelEncoder        - Scikit-learn LabelEncoder used to encoding classes names  
    n:          int                 - number of most relevant features used

    Returns:
    A dictionary that maps name of class to list of n most relevant features
    """
    most_important: defaultdict = defaultdict(DataFrame)
    classes_names = encoder.classes_
    feature_names = v.get_feature_names()
    columns = ["Feature", "Weight"]
    if len(classes_names) == 2:

        indices = clf.coef_[0].argsort()[-n:][::-1]
        data = []
        for index in indices:
            data.append([feature_names[index], clf.coef_[0][index]])
        class_name = encoder.inverse_transform([0])[0]
        most_important[class_name] = DataFrame(
            data, columns=columns, index=range(1, n + 1)
        )

        indices = (-clf.coef_[0]).argsort()[-n:][::-1]
        data = []
        for index in indices:
            data.append([feature_names[index], (-clf.coef_[0])[index]])
        class_name = encoder.inverse_transform([1])[0]
        most_important[class_name] = DataFrame(
            data, columns=columns, index=range(1, n + 1)
        )

    elif len(classes_names) > 2:

        for i in range(len(classes_names)):
            indices = clf.coef_[i].argsort()[-n:][::-1]  # n largest elements
            data = []
            for index in indices:
                data.append([feature_names[index], clf.coef_[i][index]])
            class_name = encoder.inverse_transform([i])[0]
            most_important[class_name] = DataFrame(
                data, columns=columns, index=range(1, n + 1)
            )
    return dict(most_important)


def cleaning(*, path: Path, extension: str, affix: str) -> None:
    """Deletes all files in a Path with certain extension and name termination.
    
    Parameters:
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


def clean_files():
    """Deletes all files created during processing of corpora no longer needed during experiments."""
    CORPORA = Path(fr"{ROOT}/Corpora/Proc_Quixote")
    if CORPORA.exists():
        cleaning(path=CORPORA, extension=".txt", affix="proc")
        CORPORA.rmdir()

    CORPORA = Path(fr"{ROOT}/Corpora/Proc_Ibsen")
    if CORPORA.exists():
        cleaning(path=CORPORA, extension=".txt", affix="proc")
        CORPORA.rmdir()

    SN_FOLDER = Path(fr"{ROOT}/auxfiles/txt")
    cleaning(path=(SN_FOLDER / "Quixote"), extension=".txt", affix="sn")
    cleaning(path=(SN_FOLDER / "Ibsen"), extension=".txt", affix="sn")


def clean_example():
    """Deletes all files created during teh example run in `analysis` submodule."""
    JSON_FOLDER = Path(fr"{ROOT}/auxfiles/json/")
    cleaning(path=JSON_FOLDER, extension=".json", affix="trash")

    clean_files()


if __name__ == "__main__":
    pass
