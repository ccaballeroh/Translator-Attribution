# Translator Attribution

**NOTE:** I had to do several last minute changes and haven't updated the comments and docstrings

## Introduction

To accurately attribute a piece of text to a certain author (i.e., Authorship Attribution) is a well-established task in computational linguistics. However, to accurately attribute a translated document to its translator (we might call it Translatorship Attribution) is a seemingly understudied task. There are only a handful of articles tackling this problem, and only a subset of them reports successful results.

For my thesis project, I want to attribute the translator of a text to its translator using linguitics-inspired features. The corpora comprises works by Norwegian playwright, Henrik Ibsen, and Spanish writer, Miguel de Cervantes. For the Ibsen corpus, there are parallel works translated by two people, whereas the Cervantes corpus is composed of three translations.

This repository holds all the code and raw files necessary to replicate the results for my master's project. All the texts are in the `Corpora` folder in raw format. 

## Important!

- Clone or download this repository in a Google Drive folder named `Translator-Attribution`.
- You can execute now the Notebooks on Colab without installing anything&mdash;starting with [01Processing](./01Processing.ipynb).

### To run locally

- Use conda to create a new environment called `translator-attribution`.

  ```
  >> conda env create --file translator-attribution.yml
  ```
- Now, you can, in that new environment (use `conda activate translator-attribution`), open the Notebooks in Jupyter&mdash;starting with [01Processing](./01Processing.ipynb).
