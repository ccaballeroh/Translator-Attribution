import spacy
from collections import defaultdict
from spacy.tokens import Span
from spacy.lang.en import English
from pathlib import Path
import json
from typing import Dict, DefaultDict
import re

__all__ = [
    "MyDoc",
    "extract_syntactic_ngrams",
    "save_dataset_to_json",
    "get_dataset_from_json",
]


JSON_FOLDER = Path(r"./auxfiles/json/")
OUTPUT_FOLDER = Path(r"./auxfiles/json/")
MFILE = Path(r"markersList.json")


class MyDoc(object):
    def __init__(self, filename: Path, nlp):
        self.__file = filename
        self.nlp = nlp
        self.text = filename.read_text()
        self.__doc = self.nlp(self.text)
        self.__translator = filename.name.split("_")[0]

    @property
    def file(self) -> Path:
        return self.__file

    @property
    def filename(self) -> str:
        return self.__file.name

    @property
    def doc(self):
        return self.__doc

    @property
    def translator(self) -> str:
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

    def cohesive(self, punct=False):
        """Returns cohesive markers values from a spaCy doc.

        The function takes as inputs a processed document,
        a matcher object and a flag to take into account the punctuation
        surrounding the marker.

        :doc      'doc'           - spaCy doc object of the document
        :matcher  'PhraseMatcher' - spaCy matcher with the markers to match
        :extended 'Boolean'       - Flag to take into account punctuation

        Returns a dictionary with markers as keys and counts as values.
        """
        doc = self.doc
        matcher = self._marker_matcher()
        features = defaultdict(int)
        matches = matcher(doc)
        spans = [Span(doc, start, end) for match_id, start, end in matches]
        if punct:
            spans = self._extended_spans(spans)
        for string in (span.text.lower() for span in spans):
            features[string] += 1
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

    def _extended_spans(self, spans):
        """Returns Spans with surrounding punctuation.

        The function takes a list of Span spaCy objects, and
        a document pipelined with spaCy.

        :spans 'List' - List with Span objects where to look for
                        surrounding punctuation
        :doc   'doc'  - spaCy doc object where to look for punctuation

        Returns a list of extended Spans if the Span was surrounded by punctuation.
        """
        doc = self.doc
        extended_spans = []
        for span in spans:
            start, end = span.start, span.end
            previous_token = doc[start - 1]
            following_token = (
                doc[end] if end < len(doc) else doc[end - 1]
            )  # Atención c/ el índice
            if previous_token.is_punct:
                start = start - 1 if (start - 1) > 1 else start
            if following_token.is_punct:
                end = end + 1
            extended_spans.append(Span(doc, start, end))
        return extended_spans

    def _marker_matcher(self, markersfile: Path = MFILE, folder: Path = JSON_FOLDER):
        """Returns a spaCy's PhraseMatcher object.

        The function takes a nlp pipe, a filename to read from the words
        to match, and optionally the folder where to find the file.

        :nlp          'spaCy pipe object' - spaCy pipe with at leat a tokenizer
        :markersfile  'Str'               - String with filename to read from.

        Returns a PhraseMatcher object.
        """
        from spacy.matcher import PhraseMatcher

        nlp = self.nlp
        with open(folder / markersfile, "r") as f:
            MARKERS = json.loads(f.read())
        markers_pattern = list(nlp.pipe(MARKERS))
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        matcher.add("MARKERS", markers_pattern)
        return matcher


def save_dataset_to_json(featureset, jsonfilename, outputfolder=OUTPUT_FOLDER):
    """Writes to json file featureset.

    The feature set comprises a list of lists. Each list contains
    a dictionary of feature names -> values and a string with the
    translator's name.

    :featureset    'List'   - [[{'feature':value},'translator'], ...]
    :jsonfilename  'String' - Filename without .json extension
    :outputfolder  'Path'   - path to save file

    Returns None.
    """
    json_str = json.dumps(featureset)
    with open(outputfolder / (jsonfilename + ".json"), "w") as f:
        f.write(json_str)
    print("file saved")
    return None


def get_dataset_from_json(jsonfilename):
    """Loads dataset from json file.

    For each document, the json file contains dictionary
    with featurename -> value and string with translator's
    name.

    :jsonfilename  'Path' - name of file to load with full path
    
    Returns a tuple of lists.
    First is list of dictionary of features.
    Second is list of strings with translators names"""
    with open(jsonfilename, "r") as f:
        dataset = json.loads(f.read())
    X_dict = [features for features, _ in dataset]
    y_str = [translator for _, translator in dataset]
    return X_dict, y_str


def _example():
    # nlp = English()
    nlp = spacy.load("en_core_web_sm")
    filename = Path.cwd() / "Corpora" / "Proc_Quixote" / "Jarvis_p1_ch1_proc.txt"

    return MyDoc(filename, nlp)


if __name__ == "__main__":
    print(
        f"""
    This is an example...
    
    We're going to process the file *Jarvis_pq_ch1_proc.txt*
    located in {Path.cwd()/'Corpora'/'Proc_Quixote'} 
    """
    )
    doc = _example()
    print(
        f"""
    Now, the document *doc* is of type {type(doc)}.
    We can inspect its properties such as doc.filename = {doc.filename},
    its translator doc.translator = {doc.translator}, and doc.file =
    {doc.file} of type {type(doc.file)}
    """
    )
    d1, d2 = doc.n_grams(n=3, punct=True), doc.n_gramsPOS(n=3, punct=False)
    input("...")
    print(
        f"""
    We can now extract the features. For example, we can extract trigrams with
    punctuation along with POS trigrams without punctuation...
    """
    )
    input("The first ones are stored in the dictionary d1...")
    print(
        f"""
    {d1}
    """
    )
    input("The second ones in variable d2...")
    print(
        f"""
    {d2}
    """
    )
    d1.update(d2)
    input("We can combine them d1.update(d2)...")
    print(
        f"""
    {d1}
    """
    )
    input("...")
    print(
        f"""
    And save the result to disk save_dataset_to_json([(d1,doc.translator], "trash")
    """
    )
    save_dataset_to_json([(d1, doc.translator)], "trash")
    print(
        f"""
    We can retrive them again and save them into variables ready to ingest
    to a machine learning model X, y = get_dataset_from_json(JSON_FOLDER/'trash.json')
    """
    )
    X, y = get_dataset_from_json(JSON_FOLDER / "trash.json")
