# Translator Attribution

## Introduction

To accurately attribute a piece of text to a certain author (i.e., Authorship Attribution) is a well-established task in computational linguistics. However, to accurately attribute a translated document to its translator (we might call it Translatorship Attribution) is a seemingly understudied task. There are only a handful of articles tackling this problem, and only a subset of them reports successful results.

For my thesis project, I want to attribute the translator of a text to its translator using linguitics-inspired features. The corpora comprises works by Norwegian playwright, Henrik Ibsen, and Spanish writer, Miguel de Cervantes. For the Ibsen corpus, there are parallel works translated by two people, whereas the Cervantes corpus is composed of three translations.

This repository holds all the code and raw files necessary to replicate the results for my master's project. All the texts are in the `Corpora` folder in raw format. 

## Important!

- Clone or download this repository in a Google Drive folder named `Translator-Attribution`.
- Use conda to create a new environment called `translator-attribution`.

  ```
  >> conda env create --file translator-attribution.yml
  ```
- Now, you can in that new environment (use `conda activate translator-attribution`) either open [01Processing](./01Processing.ipynb) locally or run it in Google Colab.
- You can run [02Experiments](./02Experiments.ipynb) in Google Colab preferentably or locally if you have a powerful enough computer.
- [03Most_important_features](./03Most_important_features.ipynb) extracts the most relevant features for each translator and generates loads of figures and tables (csv, html, and LaTeX) with the results. Although, the number of figures and files generated is above 800 hundred (for all the combinations possible), the folder weighs less than 9 MB.
