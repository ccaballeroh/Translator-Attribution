"""This module contains the class definition of `MyDoc` and two functions useful
for saving and retrieving feature sets to and from disk.

The feature sets are a bag-of-words model of any combination of:
    - word n-grams with or without punctuation
    - POS n-grams with or without punctuation
    - syntactic n-grams with n in [2, 7]
    - cohesive markers with or without punctuation

They are returned as a dictionary of {feature : counts, ...} easily serialized to a
json file when stored in lists.
"""

from collections import defaultdict
from pathlib import Path
from spacy.tokens.doc import Doc
from spacy.lang.en import English
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span
from spacy.tokens.token import Token
from typing import Dict, DefaultDict, Generator, List, IO, Tuple
import json
import re
import spacy
import subprocess


__all__ = [
    "MyDoc",
    "save_dataset_to_json",
    "get_dataset_from_json",
]

TXT_FOLDER = Path(r"./auxfiles/txt/")
JSON_FOLDER = Path(r"./auxfiles/json/")
MFILE = Path(r"markersList.json")


class MyDoc:
    """
    MyDoc class represents a document text processed by a spaCy language model. It has methods
    for returning feature sets.
    
    The feature sets are a bag-of-words model of any combination of:
    - word n-grams with or without punctuation
    - POS n-grams with n in {2, 3} with or without punctuation
    - syntactic n-grams with n in {2, 3}
    - cohesive markers with or without punctuation

    Attributes:
    - author: str 
        author of the text file
    - doc: spacy.tokens.doc.Doc 
        processed spaCy doc
    - file: Path 
        Path object of processed file
    - filename: str 
        file name 
    - matcher: PhraseMatcher 
        spaCy PhraseMatcher for finding cohesive markers
    - nlp: English 
        spaCy English language model
    - text: str 
        plain text representation of the processed text file
    - translator: str 
        translator of the text file

    Methods:
    - n_grams(*, n: int, punct: bool, pos: bool) -> Dict[str, int] 
        Returns a dictionary of n-grams and their counts.
        Can include punctuation if `punct` is True and be
        POS n-grams if pos flag set to True 
    - n_grams_syntactic(*, n: int) -> Dict[str, int] 
        n in {2, 3}, returns a dictionary of syntactic n-grams and their counts.
    - cohesive(*, punct: bool) -> Dict[str, int]
        Returns a dictionary of cohesive markers.
        Can include punctuation if `punct` is True.
    """

    def __init__(
        self,
        filename: Path,
        nlp: English,
        markersfile: Path = MFILE,
        folder: Path = JSON_FOLDER,
    ):
        """
        Parameters:
        filename:    Path    - Path object to file to read
        nlp:         English - spaCy language model to process file
        markersfile: Path    - Path object to file with cohesive markers
        folder:      Path    - Path to directory where JSON file with markers is
        """
        self.__file = filename
        self.nlp = nlp
        self.text = filename.read_text()
        self.matcher = self._marker_matcher(markersfile, folder)
        self.__doc = self.nlp(self.text)
        self.__translator = filename.name.split("_")[0]

    @property
    def file(self) -> Path:
        """Returns Path object to text file."""
        return self.__file

    @property
    def filename(self) -> str:
        """Returns name of the text file."""
        return self.__file.name

    @property
    def doc(self) -> Doc:
        """Returns spaCy doc object of the text file."""
        return self.__doc

    @property
    def translator(self) -> str:
        """Returns translator of text file."""
        return self.__translator

    @property
    def author(self) -> str:
        """Returns author of text file."""
        return self.file.parts[-2].split("_")[1]

    def __repr__(self) -> str:
        """Returns representation."""
        return str(self.doc)

    def n_grams(
        self, *, n: int, punct: bool = False, pos: bool = False
    ) -> Dict[str, int]:
        """Returns bag-of-words model of n-grams or POS n-grams with and without punctuation.

        The n-grams can include or omit the punctuation marks depending on the
        value of the parameter punct. Both parameters are keyword-only.

        Parameters:
        n:      int  - value of n in n-grams
        punct:  bool - flag to consider or not punctuation
        pos:    bool - option for "pos" n-grams 

        Returns:
        A dictionary of POS n-grams and their counts
        """
        assert isinstance(n, int) and n > 0, f"n positive integer and n={n} is given."
        assert isinstance(punct, bool), f"punct must be Boolean and {punct} is given."
        assert isinstance(pos, bool), f"pos must be boolean and {pos} is given."
        features: DefaultDict[str, int] = defaultdict(int)
        if pos:
            ngrams_str = self._ngrams_pos(n=n, punct=punct)
            strings = (" ".join(pos for pos in ngram) for ngram in ngrams_str)
        else:
            ngrams_tkns = self._ngrams_tokens(n=n, punct=punct)
            strings = (
                " ".join(token.text.lower() for token in ngram) for ngram in ngrams_tkns
            )
        for string in strings:
            features[string] += 1
        return dict(features)

    def cohesive(self, *, punct: bool = False) -> Dict[str, int]:
        """Returns cohesive markers values.

        It process the document using a spaCy PhraseMatcher to
        finde the cohesive markers given in a list.

        Parameters:
        punct: bool - Flag to take into account punctuation.

        Returns:
        Dictionary with markers as keys and counts as values.
        """
        doc = self.doc
        matcher = self.matcher
        features: DefaultDict[str, int] = defaultdict(int)
        matches = matcher(doc)
        spans = [Span(doc, start, end) for match_id, start, end in matches]
        if punct:
            spans = self._extended_spans(spans)
        for string in (span.text.lower() for span in spans):
            features[string] += 1
        return dict(features)

    def n_grams_syntactic(
        self, *, n: int = 2, minimum: int = 2, maximum: int = 3
    ) -> Dict[str, int]:
        """Returns bag-of-words model of syntactic n-grams.

        It calls helper functions to generate a file with Stanford output
        from the analysis of the document performed with spaCy. This file with
        Stanford format is used by another script to extract syntactic n-grams.
        That third-party file can extract syntactic n-grams up to a maximum n
        of 7.

        Parameters:
        n:       int - value of n in n-grams
        mimimum: int - minimum n to extract syntactic n-grams
        maximum: int - maximim n to extract syntactic n-grams 

        Returns:
        A dictionary of syntactic n-grams their counts
        """
        assert isinstance(n, int)
        assert isinstance(minimum, int)
        assert isinstance(maximum, int)
        assert minimum >= 2 and maximum <= 7, f"minimum = 2 and maximum = 7"
        assert (
            n >= minimum and n <= maximum
        ), f"n must be in [{minimum},{maximum}], n={n} given."
        author = self.author
        FOLDER = TXT_FOLDER / author
        sn_file = self.file.stem + "_sn" + self.file.suffix
        filename = FOLDER / sn_file
        if not filename.exists() or self._syntactic_features(n=n) == {}:
            print("Generating parsed and sn...")
            self._generate_parsed()  # with stanford format
            self._sn_generation(minimum=minimum, maximum=maximum)
        return self._syntactic_features(n=n)

    # helper functions

    def _generate_parsed(self) -> None:
        """Creates a file with output Stanford-like for extracting syntactic n-grams.

        The file is created with an "_parsed" appended in the end.

        Parameters:
        None

        Returns:
        None        
        """
        author = self.author
        FOLDER = TXT_FOLDER / author
        doc = self.doc
        path = FOLDER / (self.file.stem + "_parsed.txt")
        with open(path, mode="w", encoding="UTF-8") as f:
            for sentence in doc.sents:
                output_Stanford(sentence, f)
                print("", file=f)
        return None

    def _sn_generation(self, *, minimum: int, maximum: int) -> None:
        """Extracts syntactic n-grams from file with stanford format.

        It calls a third-party script for extracting syntactic n-grams
        with n being minimum 2 and maximum 7
        """
        assert isinstance(minimum, int)
        assert isinstance(maximum, int)
        assert minimum >= 2
        assert maximum <= 7
        assert minimum < maximum
        author = self.author
        FOLDER = TXT_FOLDER / author
        filename = self.file.stem + "_parsed.txt"
        assert (FOLDER / filename).exists(), "Parsed file {filename} not found"
        # subprocess that calls third party script
        command = subprocess.Popen(
            [
                "python",
                "./helper/sn_grams3.py",
                "./auxfiles/txt/%s/%s" % (author, filename),
                "./auxfiles/txt/%s/%ssn.txt" % (author, filename[:-10]),
                "%d" % minimum,  # minimum n
                "%d" % maximum,  # maximum n
                "5",
                "0",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        stdout, stderr = command.communicate()
        (FOLDER / filename).unlink()  # deletes stanford format file
        print(f"Deleted {FOLDER / filename}")
        return None

    def _syntactic_features(self, *, n: int) -> Dict[str, int]:
        """Returns syntactic ngrams values from text file with analysis.
        
        The function takes as the value of n which can go from 2 to 7.
        
        Parameters:
        n: int - n for syntactic n-grams
        
        Returns:
        Dictionary with syntactic n-grams as keys and counts as values.
        """
        assert isinstance(n, int)
        assert n >= 2 and n <= 7
        author = self.author
        FOLDER = TXT_FOLDER / author
        sn_file = self.file.stem + "_sn" + self.file.suffix
        filename = FOLDER / sn_file
        sn_pattern = r"(\w+\[.+\])\s+(\d)"  # [word[more[more]]] 1
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

    def _ngrams_tokens(
        self, *, n: int, punct: bool
    ) -> Generator[List[Token], None, None]:
        """Returns a generator of lists of n-grams of spaCy tokens.

        The function takes a value for n and a flag to consider punctuation.
        
        Parameters:
        n:     int  - value for n
        punct: bool - flag for punctuation
        
        Returns:
        Generator of lists of slices of spaCy tokens.
        """
        assert isinstance(n, int)
        assert isinstance(punct, bool)
        assert n > 0
        doc = self.doc
        if punct:
            tokens = [token for token in doc]
        else:
            tokens = [token for token in doc if token.pos_ != "PUNCT"]
        return (tokens[i : i + n] for i in range(len(tokens) + 1 - n))

    def _ngrams_pos(self, n: int, punct: bool) -> Generator[List[str], None, None]:
        """Returns a generator of lists of strings of POS n-grams.

        The function takes a value for n and a flag to consider punctuation.
        
        Parameters:
        n:     int  - value for n
        punct: bool - flag for punctuation
        
        Returns:
        Generator of lists of strings of POS n-grams.
        """
        assert isinstance(n, int)
        assert isinstance(punct, bool)
        assert n > 0
        doc = self.doc
        if punct:
            pos = [token.text if token.pos_ == "PUNCT" else token.pos_ for token in doc]
        else:
            pos = [token.pos_ for token in doc if token.pos_ != "PUNCT"]
        return (pos[i : i + n] for i in range(len(pos) + 1 - n))

    def _extended_spans(self, spans: List[Span]) -> List[Span]:
        """Returns Spans with surrounding punctuation.

        The function takes a list of Span spaCy objects, and
        a document pipelined with spaCy.

        Parameters:
        spans: List[Span] - List with Span objects where to look for
                            surrounding punctuation

        Returns:
        List of extended Spans if the Span was surrounded by punctuation.
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

    def _marker_matcher(self, markersfile: Path, folder: Path) -> PhraseMatcher:
        """Returns a spaCy's PhraseMatcher object.

        The function takes a nlp pipe, a filename to read from the words
        to match, and optionally the folder where to find the file.

        Parameters:
        markersfile:    Path - Path object with filename to read from.
        folder:         Path - Path object with directory where to read from the file 
        
        Returns:
        PhraseMatcher object.
        """
        nlp = self.nlp
        path = folder / markersfile
        with open(path, mode="r") as f:
            MARKERS = json.loads(f.read())
        markers_pattern = list(nlp.pipe(MARKERS))
        matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
        matcher.add("MARKERS", markers_pattern)
        return matcher


def save_dataset_to_json(
    featureset: List[Tuple[Dict[str, int], str]],
    jsonfilename: str,
    outputfolder: Path = JSON_FOLDER,
) -> None:
    """Writes to json file featureset.

    The feature set comprises a list of tuples. Each tuple contains
    a dictionary of feature names -> values and a string with the
    translator's name.

    Parameters:
    featureset:     List[Tuple[Dict[str, int], str]]    - [({'feature':value, ...},'translator'), ...]
    jsonfilename:   str                                 - Filename without .json extension
    outputfolder:   Path                                - Path object where to save file

    Returns:
    None
    """
    json_str = json.dumps(featureset)
    path = outputfolder / (jsonfilename + ".json")
    with open(path, mode="w") as f:
        f.write(json_str)
    print("file saved")
    return None


def get_dataset_from_json(jsonfilename: Path) -> Tuple[List[Dict[str, int]], List[str]]:
    """Loads dataset from json file.

    For each document, the json file contains dictionary
    with featurename -> value and string with translator's
    name.

    Parameters:
    jsonfilename: Path - Path object of file to load
    
    Returns:
    Tuple of lists. 
    First is list of dictionary of features.
    Second is list of strings with translators names
    """
    with open(jsonfilename, mode="r") as f:
        dataset = json.loads(f.read())
    X_dict = [features for features, _ in dataset]
    y_str = [translator for _, translator in dataset]
    return X_dict, y_str


def get_root(sentence: Span) -> Token:
    """Returns the root token for a span.

    The span must be a sentence from a processed spaCy Doc.

    Parameters:
    sentence: Span - The sentence where to extract the root

    Returns:
    The root token.
    """
    for token in sentence:
        if token.dep_ == "ROOT":
            root_node = token
    return root_node


def output_Stanford(sentence: Span, f: IO) -> None:
    """Prints to a file the Stanford format of a given sentence.

    Parameters:
    sentence:   Span    - sentence to format
    f:          IO      - file where to write

    Returns:
    None
    """
    root = get_root(sentence)
    for token in sentence:
        if token == root:
            print(f"root(ROOT-0, {token.text}-{token.i+1})", file=f)
        else:
            print(
                f"{token.dep_}({token.head}-{token.head.i+1}, {token.text}-{token.i+1})",
                file=f,
            )
    return None


def _example() -> MyDoc:
    # nlp = English()
    nlp = spacy.load("en_core_web_sm")
    filename: Path = Path.cwd() / "Corpora" / "Proc_Quixote" / "Jarvis_p1_ch1_proc.txt"
    if not filename.exists():
        print("Preprocessing Quixote texts...")
        from helper.preprocessing import quixote

        quixote()
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
    input("...")
    print(
        f"""
    We can now extract the features. For example, we can extract trigrams with
    punctuation along with POS trigrams without punctuation:
    
    d1, d2 = doc.n_grams(n=3, punct=True), doc.n_gramsPOS(n=3, punct=False)
    
    """
    )
    d1, d2 = doc.n_grams(n=3, punct=True), doc.n_grams(n=3, punct=False, pos=True)
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
    And save the result to disk save_dataset_to_json([(d1,doc.translator)], "trash")
    """
    )
    save_dataset_to_json([(d1, doc.translator)], "trash")
    print(
        f"""
    We can retrieve them again and save them into variables ready to ingest
    to a machine learning model X, y = get_dataset_from_json(JSON_FOLDER/'trash.json')
    """
    )
    from helper.utils import clean_example

    X, y = get_dataset_from_json(JSON_FOLDER / "trash.json")

    print(f"Deleting all files related to this example...")
    clean_example()
