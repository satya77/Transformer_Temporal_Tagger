"""
Script to tag the benchmark datasest using heideltime python package and output them in the format of tempeval script
"""

import os
from argparse import ArgumentParser
from tqdm import tqdm

from bs4 import BeautifulSoup
from python_heideltime.python_heideltime import Heideltime


def remove_header_and_footer_from_heideltime_doc(doc):
    """
    HeidelTime creates a header and footer. This function removes them.
    @param doc: The document header and footer are removed from
    @return: The document without header and footer
    """
    doc = doc.replace('<?xml version="1.0"?>\n<!DOCTYPE TimeML SYSTEM "TimeML.dtd">\n<TimeML>\n', '')
    doc = doc.replace('\n</TimeML>\n\n', '')
    return doc


def get_text_and_annotations_and_date(in_fp):
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
        # if there is no value set it to null
        except KeyError:
            value = "null"
        annotations.append((begin, end, timex.attrs["type"], value))

    text = content.text

    return text, annotations, date


def process_file(in_fp, heideltime_parser, tweets):
    """
    Process a file with heideltime and add the header and footer of the original file to it
    :param in_fp: input file
    :param heideltime_parser: insatnce of heideltime
    :param tweets: if it is the tweets daataset
    :return:
    """
    text, annotations, date = get_text_and_annotations_and_date(in_fp)
    heideltime_parser.set_document_time(date)
    heidel_xml = heideltime_parser.parse(text)
    heidel_xml = remove_header_and_footer_from_heideltime_doc(heidel_xml)

    with open(in_fp) as f:
        all_line = "\n".join(f.readlines())
    if not tweets:  # tweets has a slightly dfferent line format
        id_start = all_line.find("<TEXT>")
        beginning_text = all_line[:id_start] + "<TEXT>"
        end_text = "\n\n</TEXT>\n</TimeML>"
    else:
        id_start = all_line.find("<TEXT>")
        beginning_text = all_line[:id_start] + "<TEXT>"
        end_text = "\n</TEXT>\n</TimeML>"
        if tweets:
            end_text = "</TEXT>\n</TimeML>"

    return heidel_xml, beginning_text, end_text


def get_args():
    args = ArgumentParser()

    args.add_argument("--input_folder", type=str, default="Bert_got_a_date/data/temporal/tempeval/tempeval_test",
                      help="file from the benchmark datasset to be processed")
    args.add_argument("--output_folder", type=str, default="/results/baselines/heideltime/tempeval",
                      help="file to be processed")
    args.add_argument("--tweets", type=bool, default=False,
                      help="Is it the tweets dataset.")

    return args.parse_args()


if __name__ == "__main__":
    args = get_args()

    output_folder = args.output_folder
    input_folder = args.input_folder
    tweets = args.tweets
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_folder), exist_ok=True)

    heideltime_parser = Heideltime()
    heideltime_parser.set_document_type("NEWS")

    for fn in tqdm(sorted(os.listdir(input_folder))):
        # Only process files with the correct extension
        if fn.endswith("tml"):
            tagged_data, beginning_text, end_text = process_file(os.path.join(input_folder, fn),
                                                                 heideltime_parser,
                                                                 tweets)
            with open(os.path.join(output_folder, fn), 'w') as f:
                combined_out = beginning_text + tagged_data + end_text

                combined_out = combined_out.replace("\n\n", "\n")
                f.write(combined_out)
