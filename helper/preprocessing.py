import re
import os
from pathlib import Path

CORPORA = Path(r"./Corpora/")


def remove_numbers(text):
    """
    """
    pattern = re.compile(r"((-?\[\d+\]-?)|(-?\(\d+\)-?))")
    clean_text = pattern.sub(r" ", text)
    return clean_text


def collapse_spaces(text):
    """
    """
    pattern = re.compile(r"\s+")
    clean_text = pattern.sub(r" ", text)
    return clean_text


def remove_special(text, REPLACE):
    """
    """
    for char, subs in REPLACE.items():
        text = text.replace(char.lower(), subs)
    return text


def special_characters(INPUT_FOLDER):
    chars = set()
    for file in INPUT_FOLDER.iterdir():
        with file.open("r") as f:
            text = f.read()
        chars = chars.union(set(text))
    return list(filter(lambda char: True if ord(char) > 127 else False, chars))


def preprocess(INPUT_FOLDER, OUTPUT_FOLDER, REPLACE):
    for filename in INPUT_FOLDER.iterdir():
        with filename.open("r") as file:
            file_content = file.read()
        file_content = collapse_spaces(remove_numbers(file_content))
        file_content = remove_special(file_content, REPLACE)
        with open(
            OUTPUT_FOLDER / (filename.stem + "_proc.txt"), "w", encoding="UTF-8"
        ) as file:
            file.write(file_content)


def remove_front_back_matter(filename, output_folder):
    """Remove legal information from Project Gutenberg files.
    
    Reads the file with 'filename' in the 'input_folder' folder and
    outputs the same file with the "proc" word appended at the end
    of the filename in the 'output_folder', but without the lines at
    the beginning and at the end of the original file containing
    legal information from Project Gutenberg.
    
    :filename     'Path' - name of the file to process
    :out_folder   'Path' - name of the outout folder
    
    It returns None
    """

    lines = []
    write = False
    with open(filename, "r", encoding="UTF-8") as f:
        for line in f:
            if line.strip().startswith("*** START OF"):
                write = True
            elif line.strip().startswith("*** END OF"):
                write = False
                break
            else:
                if write:
                    lines.append(line)
                else:
                    pass

    with open(
        output_folder / (filename.stem + "_proc.txt"), "a", encoding="UTF-8"
    ) as g:
        for line in lines:
            g.write(line)
    return None


def chunks(filename, CHUNK_SIZE=5000):
    """Generator that yields the following chunk of the file.
    
    The output is a string with the following chunk size
    CHUNK_SIZE of the file 'filename' in the folder 'input folder'.
    
    :filename      'Path'    - name of file to process
    :CHUNK_SIZE    'Integer' - size of chunk
    
    yields a 'String' of size of 'CHUNK_SIZE'
    """
    SIZE = os.stat(filename).st_size  # filesize
    with open(filename, "r", encoding="UTF-8") as f:
        for _ in range(SIZE // CHUNK_SIZE):
            # reads the lines that amount to the Chunksize
            # and yields a string
            yield "".join(f.readlines(CHUNK_SIZE))


def cleaning(FOLDER):
    for file in FOLDER.iterdir():
        file.unlink()
    else:
        FOLDER.rmdir()


def quixote():
    INPUT_FOLDER = CORPORA / "Raw_Quixote/"
    OUTPUT_FOLDER = CORPORA / "Proc_Quixote/"
    if not OUTPUT_FOLDER.exists():
        OUTPUT_FOLDER.mkdir()
    special_characters(INPUT_FOLDER)
    REPLACE = dict(
        zip(
            ["à", "é", "’", "«", "ë", "“", "‘", "ù", "ü", "”", "—", "û", "â", "ç", "è"],
            ["a", "e", "'", '"', "e", '"', "'", "u", "u", '"', "-", "u", "a", "z", "e"],
        )
    )
    preprocess(INPUT_FOLDER, OUTPUT_FOLDER, REPLACE)


def ibsen():
    INPUT_FOLDER = CORPORA / "Raw_Ibsen/"
    TEMP_OUTPUT_FOLDER = CORPORA / "Proc_Ibsen_/"
    if not TEMP_OUTPUT_FOLDER.exists():
        TEMP_OUTPUT_FOLDER.mkdir()
    for file in INPUT_FOLDER.iterdir():
        remove_front_back_matter(file, TEMP_OUTPUT_FOLDER)
    for file in [
        file for file in TEMP_OUTPUT_FOLDER.iterdir() if file.suffix == ".txt"
    ]:
        str_gen = chunks(file, CHUNK_SIZE=5000)
        num = 0
        for chunk in str_gen:
            num += 1
            with open(
                TEMP_OUTPUT_FOLDER / (file.stem + f"_part{num:03}.txt"), "w"
            ) as f:
                f.write(chunk)
        file.unlink()
    INPUT_FOLDER = TEMP_OUTPUT_FOLDER
    OUTPUT_FOLDER = CORPORA / "Proc_Ibsen/"
    if not OUTPUT_FOLDER.exists():
        OUTPUT_FOLDER.mkdir()
    special_characters(INPUT_FOLDER)
    REPLACE = dict(
        zip(
            ["ê", "ü", "é", "â", "ú", "ó", "ö", "ë"],
            ["e", "u", "e", "a", "u", "o", "o", "e"],
        )
    )
    preprocess(INPUT_FOLDER, OUTPUT_FOLDER, REPLACE)
    cleaning(TEMP_OUTPUT_FOLDER)


if __name__ == "__main__":
    quixote()
    ibsen()
