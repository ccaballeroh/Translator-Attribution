from collections import defaultdict
import json
import copy
from math import log10


def save_dataset_to_json(featureset,
                         jsonfilename,
                         outputfolder=".\\auxfiles\\json\\"):
    """Writes to json file featureset.

    The feature set comprises a list of lists. Each list contains
    a dictionary of feature names -> values and a string with the
    translator's name.

    :featureset    'List'   - [[{'feature':value},'translator'], ...]
    :jsonfilename  'String' - Filename without .json extension
    :outputfolder  'String' - path to save file. Current by default

    Returns None.
    """
    json_str = json.dumps(featureset)
    with open(outputfolder + jsonfilename + ".json", "w") as f:
        f.write(json_str)
    print("file saved")
    return None


def get_dataset_from_json(jsonfilename,
                          folder=".\\auxfiles\\json\\"):
    """Loads dataset from json file.

    For each document, the json file contains dictionary
    with featurename -> value and string with translator's
    name.

    :jsonfilename  'String'  - name of file to load
    :folder        'String'  - relative path to folder where json is

    Returns a tuple of lists.
    First is list of dictionary of features.
    Second is list of strings with translators names"""
    with open(folder + jsonfilename, "r") as f:
        dataset = json.loads(f.read())
    X_dict = [features for features, _ in dataset]
    y_str  = [translator for _, translator in dataset]
    return X_dict, y_str


def get_translator(filename):
    """Get translator's name from file name.

    The filename must have the convention: '{Translator}_{Work}[_proc].txt'

    :filename  'String' - Name of file from which to fetch the translator name

    Returns a 'String' with the translator name.
    """
    return filename.split("_")[0]


def proc_texts(folder, filename, nlp, debug=False):
    """Process a raw file with a spaCy pipeline.

    The function takes the folder name, filename and nlp object

    :folder    'String'     - relative path to folder with corpus
    :filename  'String'     - name of the file to process
    :nlp       'spaCy pipe' - spaCy pipeline with at leat a tokenizer

    Returns a spacy Document object.
    """
    with open(folder + filename, "r",
              encoding="UTF-8",
             ) as f:
        raw = f.read()
    doc = nlp(raw)
    if debug:
        import datetime as dt
        t = dt.datetime.now()
        with open(
            ".\\auxfiles\\logs\\log_{}.txt".format(
                t.strftime("%Y%m%d_%H%M")), "a") as f:
            print(filename, file=f)
    return doc


def ngrams_tokens(doc, n=1, punct=False):
    """Returns n-grams of tokens.

    The function takes a spaCy doc object, a value for n, and
    a flag to consider punctuation.

    :doc   'spaCy doc object' - spaCy doc with tokens
    :n     'Int'              - value for n
    :punct 'Boolean'          - flag for punctuation

    Returns a generator of slices of spaCy tokens list.
    """
    if punct:
        tokens = [token for token in doc]
    else:
        tokens = [token for token in doc if token.pos_ != "PUNCT"]
    return (tokens[i:i+n] for i in range(len(tokens)+1-n))

def ngrams_pos(doc, n=1, punct=False):
    """Returns n-grams of POS.
    
    The function takes a spaCy doc object and, a value for n, and
    a flag to consider punctuation.
    
    :doc 'spaCy doc object' - spaCy doc with tokens
    :n     'Int'            - value for n
    :punct 'Boolean'        - flag for punctuation
    
    Returns a generator of tuples of POS strings.
    """
    if punct:
        pos = [token.text if token.pos_ == "PUNCT" else token.pos_ for token in doc]
        #pos = [token.pos_ for token in doc]
    else:
        pos = [token.pos_ for token in doc if token.pos_ != "PUNCT"]
    return (pos[i:i+n] for i in range(len(pos)+1-n))


def marker_matcher(nlp, markersfile, FOLDER=".\\auxfiles\\json\\"):
    """Returns a spaCy's PhraseMatcher object.

    The function takes a nlp pipe, a filename to read from the words
    to match, and optionally the folder where to find the file.

    :nlp          'spaCy pipe object' - spaCy pipe with at leat a tokenizer
    :markersfile  'Str'               - String with filename to read from.

    Returns a PhraseMatcher object.
    """
    from spacy.matcher import PhraseMatcher
    with open(FOLDER + markersfile, "r") as f:
        MARKERS = json.loads(f.read())
    markers_pattern = list(nlp.pipe(MARKERS))
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    matcher.add("MARKERS", markers_pattern)
    return matcher

def tf_by_translator(featureset):
    """Returns the dataset with term and term frequency.
    
    The function takes a featureset, which is a list of tuples, where
    the fist tuple is a dictionary of feature and raw counts of the
    feature per document, and the second is the label with the
    translator's name. It returns the raw values replaced by term
    frecuency taking each translator separately.
    
    :featureset    'List'   - [({'feature':value},'translator'), ...]
    
    Returns a list of tuples, with dictionary of term and term frequency,
    and translator.
    """
    dataset = copy.deepcopy(featureset)  # to not alter the original dataset
    labels = _get_labels(dataset)
    by_translator = _get_by_translator(dataset, labels)
    temp = _counts_by_translator(by_translator, labels)

    for label in labels:
        for document in by_translator[label]:
            for key in document.keys():
                document[key] /= temp[label]
    
    return dataset


def tfidf_by_translator(featureset):
    """Returns the dataset with term and tf-idf.
    
    The function takes a featureset, which is a list of tuples, where
    the fist tuple is a dictionary of feature and raw counts of the
    feature per document, and the second is the label with the
    translator's name. It returns the raw values replaced by tf-idf
    taking each translator separately. (idf = log10 (N / DF) + 1).
    
    :featureset    'List'   - [({'feature':value},'translator'), ...]
    
    Returns a list of tuples, with dictionary of term and idf, and
    translator. The idf uses a term plus one so to not discard completely
    terms used by all the translators. 
    """
    dataset = tf_by_translator(featureset)
    idf = _get_idf(featureset)
    for dictionary, translator in dataset:
        for term in dictionary.keys():
            dictionary[term] *= idf[term]
    return dataset
    

def _get_labels(dataset):
    """Returns all the translators names in the dataset.
    
    The function takes the entire dataset, comprised of a list
    of tuples, where the second element of the tuple is the
    translator's name, and returns all their names.
    
    :dataset    'List'   - [({'feature':value},'translator'), ...]
    
    It returns a set of strings with the translator's name
    """
    return {labels for dics, labels in dataset}


def _get_by_translator(dataset, labels):
    """Returns the dataset organized by translator.
    
    The function takes the entire dataset, comprised of a list
    of tuples, where the first element is a dictionary of terms and
    their raw counts (per document), and the second element of the
    tuple is the translator's name, and returns all translators names in
    a set, and the dictionaries by translator in a dictionary. 
    
    :dataset    'List'   - [({'feature':value},'translator'), ...]
    :labels         "Set"         - {"translator1", "translator2",...}
    
    It returns a tuple of set and dictionary.
    """
    by_translator = defaultdict(dict)
    for label in labels:
        by_translator[label] = [doc for doc, translator in dataset if translator == label]
    return dict(by_translator)


def _counts_by_translator(by_translator, labels):
    """Returns all the raw counts of terms used by each translator.
    
    The function takes the dataset organized by translator and a set with
    all the translators' names. It then sums the raw values for each term
    and saves it in a dictionary with the translator's name as key.
    
    :by_translator  "Dictionary"  - {"translator1": [{doc1},...], "translator2": ...}
    :labels         "Set"         - {"translator1", "translator2",...}
    
    It returns a dictionary of translator's name -> total count key->value pairs.
    """
    temp = defaultdict(int)
    for label in labels:
        documents = by_translator[label]
        for dictionary in documents:
            for term in dictionary.keys():
                temp[label] += dictionary[term]
    return dict(temp)

def _get_vocabs(by_translator, labels):
    """Returns the entire vocabulary of terms and the vocabulary per translator.
    
    The function takes as input the dataset organized by translator and the set
    of labels with the translators' names and returns the set of the whole vocabulary
    of terms used in the corpus and a dictionary organized by translator and a set
    of the terms used by that translator.
    
    :by_translator  "Dictionary"  - {"translator1": [{doc1},...], "translator2": ...}
    :labels         "Set"         - {"translator1", "translator2",...}
    
    Returns a tuple of set of vocabulary and dictionary of translator's name
    and set of vocabulary of termns key->value pairs.
    """
    vocab_by_trans = defaultdict(set)
    vocab = set()
    for label in labels:
        documents = by_translator[label]
        for dictionary in documents:
            vocab_by_trans[label] = set(dictionary.keys()).union(vocab_by_trans[label])
        vocab = vocab.union(vocab_by_trans[label])
    return vocab, dict(vocab_by_trans)


def _get_idf(featureset):
    """Returns a dictionary of terms and their idf (log10 (N/DF) + 1).
    
    The function takes a list of tuples, where the first element of the tuple
    is a dictionary of terms and their raw counts in a document, and the second
    element of the tuple is a translator's name. Each element of the list is for a
    document in a corpus of translated texts. It returns a dictionary with idf per
    term in the corpus. Taking all the texts of a single translator as a whole document.
    The value is calculated as log10 (N/DF) + 1, where N is the number of translators
    in the corpus and DF is the number of translators who use a particular term. The
    term plus one is to not discard a completely a term used by all the translators.
    
    :featureset    'List'   - [({'feature':value},'translator'), ...]
    
    It returns a dictionary with the term as key and its idf as value.    
    """
    dataset = copy.deepcopy(featureset)
    labels = _get_labels(dataset)
    by_translator = _get_by_translator(dataset, labels)
    global_vocab, vocab_by_translator = _get_vocabs(by_translator, labels)
    idf = defaultdict(float)
    N = len(labels)  # Number of different translators
    for term in global_vocab:
        DF = 0  # Variable where to save the number of translators who use a term
        for label in labels:
            if term in vocab_by_translator[label]:
                DF += 1
        idf[term] = log10(N/DF) + 1
    return dict(idf)
