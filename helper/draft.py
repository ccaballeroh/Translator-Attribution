import spacy
from collections import defaultdict
from spacy.tokens import Span
from pathlib import Path
from typing import Dict, DefaultDict
import re


class MyDoc(object):
    def __init__(self, filename, nlp):
        with open(filename, "r") as f:
            self.text = f.read()
        self.nlp = nlp
        self.__doc = self.nlp(self.text)
        self.__translator = filename.name.split("_")[0]

    @property
    def doc(self):
        return self.__doc

    @property
    def translator(self):
        return self.__translator

    def __repr__(self):
        return str(self.doc)

    def n_grams(self, n: int, punct: bool = False) -> Dict[str, int]:
        doc = self.doc
        features: DefaultDict[str, int] = defaultdict(int)
        ngrams = self._ngrams_tokens(n=n, punct=punct)
        if n == 1:
            strings_words = [token.text.lower() for token, in ngrams]
            for word in strings_words:
                features[word] += 1
        elif n == 2:
            strings_bigrams = [
                " ".join([token1.text.lower(), token2.text.lower()])
                for token1, token2 in ngrams
            ]
            for bigram in strings_bigrams:
                features[bigram] += 1
        elif n == 3:
            strings_trigrams = [
                " ".join(
                    [token1.text.lower(), token2.text.lower(), token3.text.lower()]
                )
                for token1, token2, token3 in ngrams
            ]
            for trigram in strings_trigrams:
                features[trigram] += 1
        return dict(features)

    def n_gramsPOS(self, n: int, punct: bool = False) -> Dict[str, int]:
        features: DefaultDict[str, int] = defaultdict(int)
        ngrams = self._ngrams_pos(n=n, punct=punct)
        if n == 2:
            strings_bigrams = [" ".join([pos1, pos2]) for pos1, pos2 in ngrams]
            for bigram in strings_bigrams:
                features[bigram] += 1
        elif n == 3:
            strings_trigrams = [
                " ".join([pos1, pos2, pos3]) for pos1, pos2, pos3 in ngrams
            ]
            for trigram in strings_trigrams:
                features[trigram] += 1
        return dict(features)

    def _ngrams_tokens(self, n=1, punct=False):
        """Returns n-grams of tokens.
        The function takes a spaCy doc object, a value for n, and
        a flag to consider punctuation.
        :doc   'spaCy doc object' - spaCy doc with tokens
        :n     'Int'              - value for n
        :punct 'Boolean'          - flag for punctuation
        Returns a generator of slices of spaCy tokens list.
        """
        doc = self.doc
        if punct:
            tokens = [token for token in doc]
        else:
            tokens = [token for token in doc if token.pos_ != "PUNCT"]
        return (tokens[i : i + n] for i in range(len(tokens) + 1 - n))

    def _ngrams_pos(self, n=1, punct=False):
        """Returns n-grams of POS.
        
        The function takes a spaCy doc object and, a value for n, and
        a flag to consider punctuation.
        
        :doc 'spaCy doc object' - spaCy doc with tokens
        :n     'Int'            - value for n
        :punct 'Boolean'        - flag for punctuation
        
        Returns a generator of tuples of POS strings.
        """
        doc = self.doc
        if punct:
            pos = [token.text if token.pos_ == "PUNCT" else token.pos_ for token in doc]
        else:
            pos = [token.pos_ for token in doc if token.pos_ != "PUNCT"]
        return (pos[i : i + n] for i in range(len(pos) + 1 - n))


def main():
    print("main...")
    nlp = spacy.load("en_core_web_sm")
    CORPUS_FOLDER = Path(r"D:/Google Drive/00Tesis/Corpora/Proc_Ibsen_final/")

    print("procesand...")
    return [
        MyDoc(file, nlp) for file in CORPUS_FOLDER.iterdir() if file.suffix == ".txt"
    ]


def main_ngrams(n, docs, punct=False):
    print(f"features...{n}")
    return [(my_doc.n_grams(n, punct=punct), my_doc.translator) for my_doc in docs]


def main_ngramsPOS(n, docs, punct=False):
    print(f"features...{n}")
    return [(my_doc.n_gramsPOS(n, punct=punct), my_doc.translator) for my_doc in docs]


if __name__ == "__main__":
    docs = main()
    example_punct = [main_ngrams(n + 1, docs, punct=True) for n in range(3)]
    example_no_punct = [main_ngrams(n + 1, docs, punct=False) for n in range(3)]
    example_punctPOS = [main_ngramsPOS(n + 1, docs, punct=True) for n in range(1, 3)]
    example_no_punctPOS = [
        main_ngramsPOS(n + 1, docs, punct=False) for n in range(1, 3)
    ]