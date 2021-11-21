"""
Script to create a subset of the original news articles tagged with heideltime.
"""
import json
import re
from tqdm import tqdm

from bs4 import BeautifulSoup


def create_new_files(file, out_output, total_sample, total_date, total_no_date):
    """
    Create new files based on the conditions of total number of samples and total number of with and without date samples
    :param file: the path to the file
    :param out_output: output path
    :param total_sample: total samples in the file
    :param total_date: samples with date
    :param total_no_date: samples without date
    :return:
    """
    total_counter = 0
    date_counter = 0
    no_date_counter = 0
    short_examples = 0
    long_examples = 0
    with open(file, 'r') as f, open(out_output, 'w') as w:
        for line in tqdm(f):
            json_line = json.loads(line)
            if total_counter < total_sample:
                # Skip paragraphs which are too long or short
                if len(json_line["tagged_text"]) > 50 and len(
                        json_line["tagged_text"]) < 3000:
                    if "<TIMEX3" in json_line["tagged_text"]:

                        soup = BeautifulSoup(json_line["tagged_text"], "lxml")

                        content = soup.findAll("body")[0].text
                        end = 0
                        for labels in soup.findAll("timex3"):
                            # Remove the unnecessary attributes
                            new_label = re.sub(' tid="t\d+"', '', str(labels))
                            new_label = re.sub(' mod="\w+"', '', new_label)
                            begin = content[end:].index(labels.text) + end
                            end = begin + len(labels.text)
                            temp = content[:begin] + new_label + content[end:]
                            json_line["tagged_text"] = temp

                        json_line["has_date"] = True
                        if date_counter < total_date:
                            date_counter += 1
                            total_counter += 1
                            w.write(str(json.dumps(json_line)) + "\n")
                    else:
                        json_line["has_date"] = False
                        if no_date_counter < total_no_date:
                            no_date_counter += 1
                            total_counter += 1
                            w.write(str(json.dumps(json_line)) + "\n")
                else:
                    if len(json_line["tagged_text"]) < 50:
                        short_examples = short_examples + 1
                    if len(json_line["tagged_text"]) > 3000:
                        long_examples = long_examples + 1

    print("too short examples<50:", short_examples)
    print("too long examples> 3000:", long_examples)
    print("total:", total_counter)
    print("containing date:", date_counter)
    print("not containing date:", no_date_counter)


if __name__ == "__main__":
    percentage_no_date = 0.2  # we want roughly 20 percent of the data to have no dates (negative samples)
    total_train_sample = 1_000_000  # total paragraphs for train
    date_train_samples = int((1 - percentage_no_date) * total_train_sample)
    no_date_train_samples = int(percentage_no_date * total_train_sample)

    total_test_sample = 50_000  # total paragraphs for test
    date_test_samples = int((1 - percentage_no_date) * total_test_sample)
    no_date_test_samples = int(percentage_no_date * total_test_sample)

    print("number of training sample with date should be:", date_train_samples)
    print("number of training sample with no date should be:", no_date_train_samples)

    train_file = "./data/temporal/tempeval_seq2seq_corrected/train/train_heideltime_corrected.json"
    test_file = "./data/temporal/tempeval_seq2seq_corrected/test/test_heideltime_corrected.json"

    output_train = "./data/temporal/tempeval_seq2seq_corrected/train/train_heideltime_subset_cleaned_1Mil_test.json"
    output_test = "./data/temporal/tempeval_seq2seq_corrected/test/test_heideltime_subset_cleaned_1Mil_test.json"

    print("processing training files:")
    create_new_files(train_file, output_train, total_train_sample, date_train_samples, no_date_train_samples)
    print("processing testing files:")
    create_new_files(test_file, output_test, total_test_sample, date_test_samples, no_date_test_samples)
