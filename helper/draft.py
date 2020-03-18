import spacy
from collections import defaultdict
from spacy.tokens import Span
from spacy.lang.en import English
from pathlib import Path
import json
from typing import Dict, DefaultDict
import re


JSON_FOLDER = Path(r"./auxfiles/json/")


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

    def cohesive(self, matcher, punct=False):
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


def marker_matcher(nlp, markersfile, FOLDER=JSON_FOLDER):
    """Returns a spaCy's PhraseMatcher object.

    The function takes a nlp pipe, a filename to read from the words
    to match, and optionally the folder where to find the file.

    :nlp          'spaCy pipe object' - spaCy pipe with at leat a tokenizer
    :markersfile  'Str'               - String with filename to read from.

    Returns a PhraseMatcher object.
    """
    from spacy.matcher import PhraseMatcher

    with open(FOLDER / markersfile, "r") as f:
        MARKERS = json.loads(f.read())
    markers_pattern = list(nlp.pipe(MARKERS))
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("MARKERS", markers_pattern)
    return matcher

if __name__ == "__main__":
    pass
