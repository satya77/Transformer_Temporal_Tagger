"""
Script to generate stats of the evaluation corpora.
We go beyond previous work by also investigating the distribution of taggings.
"""

import os
import json
from collections import Counter

import spacy
import numpy as np
from bs4 import BeautifulSoup


def get_heideltime_corpus_stats(heideltime_file, nlp):
    type_dist = {"DATE": 0, "SET": 0, "DURATION": 0, "TIME": 0}
    all_num_sentences = []
    all_num_annotations = []
    with open(heideltime_file) as f:
        json_lines = f.readlines()

    prev_id = json.loads(json_lines[0].strip("\n "))["id"]
    num_sentences = 0
    num_annotations = 0
    for line in json_lines:
        current_sample = json.loads(line.strip("\n "))
        if current_sample["id"] != prev_id:
            all_num_sentences.append(num_sentences)
            all_num_annotations.append(num_annotations)

            num_sentences = 0
            num_annotations = 0

        prev_id = current_sample["id"]

        num_sentences += get_sentence_length(current_sample["text"], nlp)
        annotations = BeautifulSoup(current_sample["tagged_text"], "lxml").findAll("timex3")
        num_annotations += len(annotations)
        for annotation in annotations:
            type_dist[annotation.attrs["type"]] += 1

    # Edge case for last sample
    all_num_sentences.append(num_sentences)
    all_num_annotations.append(num_annotations)

    print_stats(all_num_annotations, all_num_sentences, type_dist)


def get_corpus_stats(corpus_dir, nlp):
    all_num_sentences = []
    all_num_annotations = []
    all_type_dist = {"DATE": 0, "SET": 0, "DURATION": 0, "TIME": 0}
    for subdir, dirs, files in os.walk(corpus_dir):
        for fn in sorted(files):
            if not fn.endswith(file_ext):
                continue

            with open(os.path.join(subdir, fn)) as f:
                soup = BeautifulSoup(f, "lxml")

            content = soup.findAll("text")[0]
            annotations = content.findAll("timex3")

            num_sentences = get_sentence_length(content.text, nlp)
            num_annotations = len(annotations)
            type_dist = Counter([annotation.attrs["type"] for annotation in annotations])

            all_num_sentences.append(num_sentences)
            all_num_annotations.append(num_annotations)
            for key, val in type_dist.items():
                all_type_dist[key] += val

    print_stats(all_num_annotations, all_num_sentences, all_type_dist)


def get_sentence_length(text, spacy_model):
    processed = spacy_model(text)
    sentence_length = 0
    for sent in processed.sents:
        if sent.text.strip("\n "):
            sentence_length += 1

    return sentence_length


def print_stats(all_num_annotations, all_num_sentences, all_type_dist):
    print(f"Total number of files in this data set: {len(all_num_annotations)} files")
    print(f"Mean number of sentences (+/- one standard deviation) per article: {np.mean(all_num_sentences):.2f} "
          f"(+/- {np.std(all_num_sentences):.1f}) sentences")
    print(f"Median number of sentences per article: {np.median(all_num_sentences)}")
    print(f"Total number of TIMEX3 annotations: {sum(all_num_annotations)}")
    print(f"Mean number of TIMEX3 annotations (+/- one standard deviation) per article: "
          f"{np.mean(all_num_annotations):.2f} (+/- {np.std(all_num_annotations):.1f}) annotations")
    print(f"Median number of TIMEX3 annotations per article: {np.median(all_num_annotations)}")
    print(f"Distribution of tag types:\n"
          f"{all_type_dist['DATE'] / sum(all_num_annotations) * 100:.2f}% ({all_type_dist['DATE']}) \"DATE\" tags\n"
          f"{all_type_dist['SET'] / sum(all_num_annotations) * 100:.2f}% ({all_type_dist['SET']}) \"SET\" tags\n"
          f"{all_type_dist['DURATION'] / sum(all_num_annotations) * 100:.2f}% ({all_type_dist['DURATION']}) \"DURATION\" tags\n"
          f"{all_type_dist['TIME'] / sum(all_num_annotations) * 100:.2f}% ({all_type_dist['TIME']}) \"TIME\" tags")


if __name__ == '__main__':
    file_ext = ".tml"

    tempeval_train_dir = "../data/temporal/tempeval_train/"
    tempeval_test_dir = "../data/temporal/tempeval_test/"

    tweets_train_dir = "../data/temporal/tweets/trainingset/"
    tweets_test_dir = "../data/temporal/tweets/testset/"

    wikiwars_train_dir = "../data/temporal/wikiwars/trainingset/"
    wikiwars_test_dir = "../data/temporal/wikiwars/testset/"

    heideltime_train_file = "/home/salmasian/numbert/data/temporal/tempeval_seq2seq_corrected/train/train_heideltime_subset_cleaned_1Mil.json"
    heideltime_test_file = "/home/salmasian/numbert/data/temporal/tempeval_seq2seq_corrected/test/test_heideltime_subset_cleaned_1Mil.json"

    nlp = spacy.load("en_core_web_sm", disable=["ner"])

    print(f"--------------------------------------------------------")
    print("Tempeval train stats:")
    get_corpus_stats(tempeval_train_dir, nlp)
    print(f"--------------------------------------------------------")
    print("Tempeval test stats:")
    get_corpus_stats(tempeval_test_dir, nlp)
    print(f"--------------------------------------------------------")
    print("Tweets train stats:")
    get_corpus_stats(tweets_train_dir, nlp)
    print(f"--------------------------------------------------------")
    print("Tweets test stats:")
    get_corpus_stats(tweets_test_dir, nlp)
    print(f"--------------------------------------------------------")
    print("Wikiwars train stats:")
    get_corpus_stats(wikiwars_train_dir, nlp)
    print(f"--------------------------------------------------------")
    print("Wikiwars test stats:")
    get_corpus_stats(wikiwars_test_dir, nlp)
    print(f"--------------------------------------------------------")
    print("Heideltime training stats:")
    get_heideltime_corpus_stats(heideltime_train_file, nlp)
    print(f"--------------------------------------------------------")
    print("Heideltime test stats:")
    get_heideltime_corpus_stats(heideltime_test_file, nlp)







