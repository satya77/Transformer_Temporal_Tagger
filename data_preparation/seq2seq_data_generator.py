"""
Script to parse Timex3 data from the benchamark dataset into json format to be processed by the seq2seq models
"""

import json
import os
from argparse import ArgumentParser
from typing import List
from tqdm import tqdm

from bs4 import BeautifulSoup

def get_args():
    args = ArgumentParser()

    args.add_argument("--input-dir-train", type=str,
                      default="../data/temporal/tempeval/tempeval_train/TimeBank,../data/temporal/tempeval/tempeval_train/AQUAINT,../data/temporal/wikiwars/trainingset/tml,../data/temporal/tweets/trainingset/tml",
                      help="Folder containing the input files to be processed, if the files are in multiple folders, seperate by spcae.")
    args.add_argument("--output-file-train", type=str,
                      default="./data/seq2seq/train/train_mixed.json",
                      help="Defines the output file path.")

    args.add_argument("--input-dir-test", type=str,
                      default="../data/temporal/tempeval/tempeval_test",
                      help="Folder containing the input files to be processed for the test subset.")
    args.add_argument("--output-file-test", type=str,
                      default="../data/seq2seq/test/tempeval_test.json",
                      help="Defines the output file for test.")
    args.add_argument("--file-ext", type=str, default="tml",
                      help="File extensions of processed files will have to match this value.")

    return args.parse_args()


def process_file(in_fp):
    """
    Process a file to get the tags and the raw text for seq2seq generation
    :param in_fp: input file
    :return:
    """
    text, annotations, date = get_text_and_annotations_and_date(in_fp)
    new_data = []
    new_text = ""
    first_begining = 0
    for annotation in annotations:
        begin, end, type, value = annotation
        new_text = new_text + text[first_begining:begin]
        new_text = new_text + ' <timex3 type="' + type + '" value="' + value + '"> ' + text[begin:end] + ' </timex3> '
        first_begining = end

    new_text = new_text + text[first_begining:]
    # gather some statistics
    counter_with_date = 0
    counter_without_date = 0
    for txt, tag in zip(text.split("\n"), new_text.split("\n")):  # each paragraph is seperated by \n
        if len(txt) > 0:
            new_data.append(
                {"text": txt, "date": date, "tagged_text": tag})  # create a new dictionary with parallel text
            if "<timex3" in tag:
                counter_with_date = counter_with_date + 1
            else:
                counter_without_date = counter_without_date + 1

    return new_data, counter_with_date, counter_without_date


def get_text_and_annotations_and_date(in_fp) -> (str, List[str], str):
    soup = BeautifulSoup(open(in_fp), "lxml")
    date = soup.findAll("dct")[0].findAll("timex3")[0].attrs["value"]
    content = soup.findAll("text")[0]
    annotations = []
    end = 0  # Only look in the "remaining string", by saving the previous end position

    for timex in content.findAll("timex3"):
        begin = content.text[end:].index(timex.text) + end
        end = begin + len(timex.text)
        try:
            value = timex.attrs["value"]
        except:
            value = "null"
        annotations.append((begin, end, timex.attrs["type"], value))

    text = content.text

    return text, annotations, date


if __name__ == "__main__":

    args = get_args()
    os.makedirs(os.path.dirname(args.output_file_train), exist_ok=True)
    os.makedirs(os.path.dirname(args.output_file_test), exist_ok=True)
    inputs = {"train": args.input_dir_train, "test": args.input_dir_test}
    json_file = {"train": [], "test": []}

    for filename in ["train", "test"]:
        if filename == "train":
            files = args.input_dir_train.split(",")
        else:
            files = [args.input_dir_test]
        for input_dir in files:
            temp_data_counter = 0
            counter_with_date_total = 0
            counter_without_date_total = 0
            for fn in tqdm(sorted(os.listdir(input_dir))):  # gather all the files into a single json
                # Only process files with the correct extension
                if fn.endswith(args.file_ext):
                    tagged_data, counter_with_date, counter_without_date = process_file(os.path.join(input_dir, fn))
                    temp_data_counter += len(tagged_data)
                    counter_with_date_total = counter_with_date_total + counter_with_date
                    counter_without_date_total = counter_without_date_total + counter_without_date
                    json_file[filename].extend(tagged_data)
            print(
                "for file {}, we have {} paragraph, {} with date, {} without date.".format(input_dir, temp_data_counter,
                                                                                           counter_with_date_total,
                                                                                           counter_without_date_total))

    out_dict = {"train": args.output_file_train, "test": args.output_file_test}
    for filename in ["train", "test"]:
        # write the output file
        with open(out_dict[filename], 'w') as f:
            for line in json_file[filename]:
                f.write(json.dumps(line) + "\n")
