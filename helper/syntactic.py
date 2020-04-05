from collections import defaultdict
from typing import Dict, DefaultDict
from pathlib import Path
import subprocess
import re
import spacy
import pickle

FOLDER = Path(r"./auxfiles/txt/")


def syntactic_ngrams(filename: Path, n: int = 2) -> Dict[str, int]:
    """Returns syntactic ngrams values from text file with analysis.
    
    The function takes as inputs the file name, along with the value of n.
    
    :filename  'Path'    - name of text file
    :n         'Integer'   - value of n for syntactic n-grams
    
    Returns a dictionary with syntactic n-grams as keys and counts as values.
    """
    sn_pattern = r"(\w+\[.+\])\s+(\d)"
    features: DefaultDict[str, int] = defaultdict(int)
    with filename.open("r") as f:
        n_ = 0
        for line in f:
            if line.startswith("*"):
                n_ = int(line.split()[-1])
            elif (line[0].isalpha()) and (n == n_):
                matchObj = re.match(sn_pattern, line)
                if matchObj:
                    sn_gram = matchObj.group(1)
                    features[sn_gram] += 1
            else:
                pass
    return dict(features)


def syntactic_features(author, n=2):
    TXT_FOLDER = FOLDER / author
    sn_files = [
        filename
        for filename in TXT_FOLDER.iterdir()
        if str(filename.stem).endswith("sn")
    ]
    return [
        (syntactic_ngrams(filename, n=2), get_translator(filename))
        for filename in sn_files
    ]


def get_root(sentence):
    for token in sentence:
        if token.dep_ == "ROOT":
            root_node = token
    return root_node


def get_translator(filename):
    return filename.name.split("_")[0]


def output_Stanford(sentence, file):
    root = get_root(sentence)
    for token in sentence:
        if token == root:
            print(
                "root",
                "(",
                "ROOT",
                "-",
                0,
                ", ",
                token.text,
                "-",
                token.i + 1,
                ")",
                sep="",
                file=file,
            )
        else:
            print(
                token.dep_,
                "(",
                token.head,
                "-",
                token.head.i + 1,
                ", ",
                token.text,
                "-",
                token.i + 1,
                ")",
                sep="",
                file=file,
            )
    return None


def sn_generation(author):
    TXT_FOLDER = FOLDER / author
    for filename in [f.name for f in TXT_FOLDER.iterdir()]:
        command = subprocess.Popen(
            [
                "python",
                "./helper/sn_grams3.py",
                "./auxfiles/txt/%s/%s" % (author, filename),
                "./auxfiles/txt/%s/%ssn.txt" % (author, filename[:-10]),
                "2",
                "3",
                "5",
                "0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout, stderr = command.communicate()


def main(author):
    TXT_FOLDER = FOLDER / author
    PICKLE = Path(fr"./auxfiles/pickle/{author}.pickle")
    with open(PICKLE, "rb") as f:
        doc_data = f.read()
    docs = pickle.loads(doc_data)
    for my_doc in docs:
        doc = my_doc.doc
        with open(
            TXT_FOLDER / (my_doc.file.stem + "_parsed.txt"), mode="w", encoding="UTF-8",
        ) as file:
            for sentence in doc.sents:
                output_Stanford(sentence, file)
                print("", file=file)


if __name__ == "__main__":
    pass
