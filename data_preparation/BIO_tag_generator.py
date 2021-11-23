"""
Script to the json files for the seq2seq models and convert them to the BIO tag format for the staging area. This format
is close to the CoNLL-2002 shared task.
For more information on the data format: https://skimai.com/how-to-fine-tune-bert-for-named-entity-recognition-ner/
"""

from argparse import ArgumentParser
from functools import lru_cache
from bs4 import BeautifulSoup
from typing import List
from tqdm import tqdm
import os

from datasets import load_dataset
import spacy


def get_args():
    args = ArgumentParser()

    args.add_argument("--input-file-train", type=str, default="../data/seq2seq/train/tempeval_train.json",
                      help="file to be processed as training data")
    args.add_argument("--input-file-test", type=str, default="../data/seq2seq/test/tempeval_test.json",
                      help="file to be processed as test data")
    args.add_argument("--output-file-train", type=str, default="../data/BIO/train_staging/tempeval_train.txt",
                      help="Defines the output file for train")
    args.add_argument("--output-file-test", type=str, default="../data/BIO/test_staging/tempeval_test.txt",
                      help="Defines the output file for tests")

    return args.parse_args()


def process_file(in_fp, out_fp):
    date = in_fp["date"]
    text, annotations = get_text_and_annotations_and_date(in_fp["text"], in_fp["tagged_text"])
    # Process with spacy for tokenization and separation of sentences
    nlp = get_spacy_model()
    doc = nlp(text)

    with open(out_fp, "a") as f:
        f.write(f"# {date}\n")
        try:
            next_annotation = annotations.pop(0)
        # Since it is EVENT + TIMEX3 data, there might be no TIMEX3 annotations in some files.
        except IndexError:
            next_annotation = next_annotation = (len(text), len(text), "O", "O")
        # Indicator for B/I tag
        B = True

        # Process each sentence separately, to make it easy to insert newlines in between.
        for sent in doc.sents:

            # Sometimes we get unfortunately starting whitespaces, which are entire sentences
            if not sent.text.strip():
                continue

            for token in sent:
                # Skip anything that isn't actually a word
                if token.is_space or token.text == "#":
                    continue

                # if we are "beyond" the current annotation, go the next one.
                if token.idx >= next_annotation[1]:
                    try:
                        next_annotation = annotations.pop(0)
                        # Reset to B tag
                        B = True
                    # Happens when we are done with the last annotation.
                    except IndexError:
                        # just maintain a dummy annotation
                        next_annotation = (len(text), len(text), "O", "O")

                line, B = construct_line(token, next_annotation, B)
                f.write(line)

            # Extra newline between sentences
            f.write("\n")


@lru_cache(maxsize=1)
def get_spacy_model(model_name="en_core_web_md"):
    return spacy.load(model_name)


def get_text_and_annotations_and_date(content,tagged_content) -> (str, List[str], str):
    soup = BeautifulSoup(tagged_content, "lxml")
    annotations = []
    end = 0  # Only look in the "remaining string", by saving the previous end position

    for timex in soup.findAll("timex3"):
        try:
            begin = content[end:].index(timex.text.strip()) + end
            end = begin + len(timex.text.strip())
            annotations.append((begin, end, timex.attrs["type"], timex.attrs["value"]))
        except:
            continue

    # Problems with newline chars force us to use this differently.
    text = content.replace("\n", " ")
    return text, annotations


def construct_line(token, next_annotation, B) -> (str, bool):
    line = token.text
    # Is the token starting position within the next annotation?
    if token.idx >= next_annotation[0]:
        if B:
            line += f" B-{next_annotation[2]}"
            B = False
        else:
            line += f" I-{next_annotation[2]}"
        line += f" {next_annotation[3]}\n"
    else:
        line += f" O O\n"

    return line, B


if __name__ == "__main__":

    args = get_args()

    os.makedirs(os.path.dirname(args.output_file_train), exist_ok=True)
    with open(args.output_file_train, "w") as f:
        f.write("")

    os.makedirs(os.path.dirname(args.output_file_test), exist_ok=True)
    with open(args.output_file_test, "w") as f:
        f.write("")

    data_files = {"train": args.input_file_train, "eval": args.input_file_test}
    # load the json dataset
    datasets = load_dataset("json", data_files=data_files)

    val_data = datasets["eval"]
    train_data = datasets["train"]

    # process the train data
    for sample in tqdm(train_data):
        process_file(sample, args.output_file_train)
    # process the test data
    for sample in tqdm(val_data):
        process_file(sample, args.output_file_test)

