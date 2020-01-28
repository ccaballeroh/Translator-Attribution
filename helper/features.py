from .functions import ngrams_tokens, ngrams_pos
from collections import defaultdict
from spacy.tokens import Span
import re


def extract_features_unigrams(doc, punct=False):
    """Returns unigrams values from a spaCy doc.

    The function takes a processed document as input.

    :doc    'doc'     - spaCy doc object of the document
    :punct  'Boolean' - flag for punctuation

    Returns a dictionary with tokens as keys and counts as values.
    """
    features = defaultdict(int)
    unigrams = ngrams_tokens(doc, n=1, punct=punct)
    strings_words = [token.text.lower() for token, in unigrams]
    for word in strings_words:
        features[word] += 1
    return dict(features)


def extract_features_bigrams(doc, punct=False):
    """Returns bigram values from a spaCy doc.

    The function takes a processed document as input.

    :doc    'doc'      - spaCy doc object of the document
    :punct  'Boolean'  - flag for punctuation

    Returns a dictionary with bigrams of tokens as keys and counts as values.
    """
    features = defaultdict(int)
    bigrams  = ngrams_tokens(doc, n=2, punct=punct)
    strings_bigrams = [" ".join([token1.text.lower(),
                                 token2.text.lower()])
                       for token1, token2 in bigrams]
    for bigram in strings_bigrams:
        features[bigram] += 1
    return dict(features)


def extract_features_trigrams(doc, punct=False):
    """Returns bigram values from a spaCy doc.

    The function takes a processed document as input.

    :doc    'doc'      - spaCy doc object of the document
    :punct  'Boolean'  - flag for punctuation    

    Returns a dictionary with trigrams of tokens as keys and counts as values.
    """
    features = defaultdict(int)
    trigrams = ngrams_tokens(doc, n=3, punct=punct)
    strings_trigrams = [" ".join([token1.text.lower(),
                                 token2.text.lower(),
                                 token3.text.lower()])
                        for token1, token2, token3 in trigrams]
    for trigram in strings_trigrams:
        features[trigram] += 1
    return dict(features)


def extract_features_bigramsPOS(doc, punct=False):
    """Returns bigram POS values from a spaCy doc.

    The function takes a processed document as input, and a flag
    to consider punctuation.

    :doc    'doc'      - spaCy doc object of the document
    :punct  'Boolean'  - flag for punctuation 

    Returns a dictionary with POS bigrams as keys and counts as values.
    """
    features = defaultdict(int)
    trigrams = ngrams_pos(doc, n=2, punct=punct)
    strings_bigrams = [" ".join([pos1, pos2])
                        for pos1, pos2 in trigrams]
    for bigram in strings_bigrams:
        features[bigram] += 1
    return dict(features)


def extract_features_trigramsPOS(doc, punct=False):
    """Returns trigram POS values from a spaCy doc.

    The function takes a processed document as input, and a flag
    to consider punctuation.

    :doc    'doc'      - spaCy doc object of the document
    :punct  'Boolean'  - flag for punctuation 

    Returns a dictionary with POS trigrams as keys and counts as values.
    """
    features = defaultdict(int)
    trigrams = ngrams_pos(doc, n=3, punct=punct)
    strings_trigrams = [" ".join([pos1, pos2, pos3])
                        for pos1, pos2, pos3 in trigrams]
    for trigram in strings_trigrams:
        features[trigram] += 1
    return dict(features)


def extract_features_cohesive(doc, matcher, extended=False):
    """Returns cohesive markers values from a spaCy doc.

    The function takes as inputs a processed document,
    a matcher object and a flag to take into account the punctuation
    surrounding the marker.

    :doc      'doc'           - spaCy doc object of the document
    :matcher  'PhraseMatcher' - spaCy matcher with the markers to match
    :extended 'Boolean'       - Flag to take into account punctuation

    Returns a dictionary with markers as keys and counts as values.
    """
    features = defaultdict(int)
    matches = matcher(doc)
    spans = [Span(doc, start, end)
             for match_id, start, end in matches]
    if extended:
        spans = _extended_spans(spans, doc)
    for string in (span.text.lower() for span in spans):
        features[string] += 1
    return dict(features)


def extract_syntactic_ngrams(folder, filename, n = 2):
    """Returns syntactic ngrams values from text file with analysis.
    
    The function takes as inputs the folder name, and file name, along
    with value of n.
    
    :folder    'String'    - name of folder
    :filename  'String'    - name of text file
    :n         'Integer'   - value of n for syntactic n-grams
    
    Returns a dictionary with syntactic n-grams as keys and counts as values.
    """
    sn_pattern = r"(\w+\[.+\])\s+(\d)"
    features = defaultdict(int)
    with open(folder + filename, "r") as f:
        n_ = 0
        for line in f:
            if line.startswith('*'):
                n_ = int(line.split()[-1])
            elif (line[0].isalpha()) and (n == n_):
                matchObj = re.match(sn_pattern, line)
                if matchObj:
                    sn_gram = matchObj.group(1)
                    features[sn_gram] += 1
            else:
                pass
    return dict(features)


def _extended_spans(spans, doc):
    """Returns Spans with surrounding punctuation.

    The function takes a list of Span spaCy objects, and
    a document pipelined with spaCy.

    :spans 'List' - List with Span objects where to look for
                    surrounding punctuation
    :doc   'doc'  - spaCy doc object where to look for punctuation

    Returns a list of extended Spans if the Span was surrounded by punctuation.
    """
    extended_spans = []
    for span in spans:
        start, end = span.start, span.end
        previous_token   = doc[start-1]
        following_token  = doc[end] if end < len(doc) else doc[end-1]  # Atención c/ el índice
        if previous_token.is_punct:
            start = start - 1 if (start-1) > 1 else start
        if following_token.is_punct:
            end = end + 1
        extended_spans.append(Span(doc, start, end))
    return extended_spans
