"""This file must be executed in order to preprocess the files.

Folders `./Corpora/Proc_Quixote` and `./Corpora/Proc_Ibsen` will
hold the preprocessed files so Google Colab can read them from
Google Drive and proceed with the next steps.
"""

from helper import preprocessing

preprocessing.quixote()
preprocessing.ibsen()

input()
