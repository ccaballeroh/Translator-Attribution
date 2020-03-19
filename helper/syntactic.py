from collections import defaultdict
from typing import Dict, DefaultDict
from pathlib import Path
import re
import spacy
import pickle


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


def get_root(sentence):
    for token in sentence:
        if token.dep_ == "ROOT":
            root_node = token
    return root_node


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


def _run():
    for translator in ["Quixote", "Ibsen"]:
        TXT_FOLDER = Path(fr"./auxfiles/txt/{translator}/")
        PICKLE = Path(fr"./auxfiles/pickle/{translator}.pickle")
        with open(PICKLE, "rb") as f:
            doc_data = f.read()
        docs = pickle.loads(doc_data)
        for my_doc in docs:
            doc = my_doc.doc
            with open(
                TXT_FOLDER / (my_doc.file.stem + "_parsed.txt"),
                mode="w",
                encoding="UTF-8",
            ) as file:
                for sentence in doc.sents:
                    output_Stanford(sentence, file)
                    print("", file=file)


if __name__ == "__main__":
    _run()
