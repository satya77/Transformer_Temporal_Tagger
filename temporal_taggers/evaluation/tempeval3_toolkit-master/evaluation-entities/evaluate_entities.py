# Original attribution: V 1.0 Naushad UzZaman, March 24, 2012
# Python3 port: Dennis Aumiller and Satya Almasian, July 2021

# this program evaluates the performance of extracted events and temporal expressions
# input files: TimeML annotated documents
# output: performance of events and temporal expression extraction
# usage:
# to check the performance of a single file:
#          python evaluate_entities.py gold_file_path system_file_path
# to check the performace of all files in a gold folder:
#          python evaluate_entities.py gold_folder_path system_folder_path

# warning: given the input file, the systems can only add XML tags of EVENT and TIMEX3.
# This evaluation program won't work properly if extra spaces, new lines are added.

# debug = 0, prints the final performance for event and timex extraction
# debug = 0.5, prints all numbers that explains how the system got the final numbers
# debug = 1, prints all performances for all files
# debug = 1.5, prints the incorrect events/timex and features
# debug = 2, prints all relevant information to trace which entities systems are missing

# *** final score ***
# we give one score to capture the performance of event/timex extraction and their feature extraction
# 50% score is for event/timex extraction and 50% for feature extraction
# 50% of extraction is divided in to strict matching and relaxed matching equally.
# For example, if the gold annotation has "Sunday morning" and the system identifies "Sunday",
# then they will get credit in relaxed matching but not in exact matching.
# for timex, 50% of attribute is divided equally between type and value
# for event, class feature gets 25% and other features gets rest 25% equally
# for event and timex, fscore (harmonic mean beween recall and precision) is calculated
# for features, the performance is recall of features,
# i.e. number of correct features/total features in gold data (total events)
from __future__ import annotations

import os
import re
import sys
import numpy as np
import pandas as pd

from dataclasses import dataclass


def get_arg(index):
    return sys.argv[index]


@dataclass
class TimexResults:
    """
    Aggregate class for relevant Timex evaluation metrics
    """
    system_timex_count: int = 0
    gold_timex_count: int = 0
    strict_match_precision: int = 0
    relaxed_match_precision: int = 0
    strict_match_recall: int = 0
    relaxed_match_recall: int = 0
    type_match: int = 0
    value_match: int = 0
    # TODO: Find a better way to encode confusion between specific types
    confusion_matrix: pd.DataFrame = pd.DataFrame(np.zeros((5, 5), dtype=int),
                                                  ('DATE', 'SET', 'TIME', 'DURATION', 'UNDEFINED'),
                                                  ('DATE', 'SET', 'TIME', 'DURATION', 'UNDEFINED'))

    def __add__(self, other: TimexResults):
        self.system_timex_count += other.system_timex_count
        self.gold_timex_count += other.gold_timex_count
        self.strict_match_precision += other.strict_match_precision
        self.relaxed_match_precision += other.relaxed_match_precision
        self.strict_match_recall += other.strict_match_recall
        self.relaxed_match_recall += other.relaxed_match_recall
        self.type_match += other.type_match
        self.value_match += other.value_match

    def update_results(self, gold_timex, system_timex):
        # Recall computation
        if debug >= 1.5:
            print('\n\nTIMEX RECALL computation')
            print('For each timex in gold annotation, compare system annotation\n')

        for tid in gold_timex:
            if gold_timex[tid].text.strip() == '':
                continue

            if debug >= 2:
                print(f' gold annotation: {gold_timex[tid].annotation}')

            if tid in system_timex and system_timex[tid].text.strip() != '':
                self.relaxed_match_recall += 1
                match_word = '-- relaxed match'
                g = gold_timex[tid]
                s = system_timex[tid]
                if gold_timex[tid].text == system_timex[tid].text:
                    self.strict_match_recall += 1
                    match_word = '-- strict match'

                if debug >= 2:
                    print(f' system annotation: {s.annotation} {match_word}')

                results.confusion_matrix[g.type][s.type] += 1

                if g.type == s.type:
                    self.type_match += 1
                if g.value == s.value:
                    self.value_match += 1

                if debug >= 1.75:
                    if g.value != s.value:
                        print(f'  -> gold value: {g.annotation}')
                        print(f'  -> system wrong value: {s.annotation}')

            else:
                g = gold_timex[tid]
                results.confusion_matrix[g.type]['UNDEFINED'] += 1
                if debug >= 1.5:
                    print(f' gold annotation not found in system: {gold_timex[tid].annotation}')

        self.gold_timex_count = len(gold_timex)
        if self.gold_timex_count != 0:
            strict_recall = self.strict_match_recall / self.gold_timex_count
            relaxed_recall = self.relaxed_match_recall / self.gold_timex_count
        else:
            strict_recall = 0
            relaxed_recall = 0

        if debug >= 1:
            print('\nTIMEX EXTRACTION RECALL PERFORMANCE')
            print(f'Strict Recall: {strict_recall:.6f}')
            print(f'Relaxed Recall: {relaxed_recall:.6f}')
        if debug >= 2:
            print(f'total: {self.gold_timex_count}')
            print(f'strict count: {self.strict_match_recall}')
            print(f'relaxed count: {self.relaxed_match_recall}')

        if debug >= 1:
            print('\nTIMEX FEATURE EXTRACTION PERFORMANCE')
            print(f'total gold timex or total features in gold data: {self.gold_timex_count}')
            print(f'total matching timex: {self.relaxed_match_recall}')
            if self.relaxed_match_recall != 0:
                print(f'type accuracy: {self.type_match} '
                      f'performance {self.type_match / self.relaxed_match_recall:.4f}')
                print(f'value accuracy: {self.value_match} '
                      f'performance {self.value_match / self.relaxed_match_recall:.4f}')
            else:
                print('timex attribute performance:', 0)

        # Precision computation
        if debug >= 1.5:
            print('\nTIMEX PRECISION computation')
            print('For each timex in system annotation, compare gold annotation\n')

        for tid in system_timex:
            if system_timex[tid].text.strip() == '':
                continue

            if debug >= 2:
                print(f' system annotation: {system_timex[tid]}')
            if tid in gold_timex and gold_timex[tid].text.strip() != '':
                self.relaxed_match_precision += 1
                match_word = '-- relaxed match'
                g = gold_timex[tid]
                s = system_timex[tid]
                if g.text == s.text:
                    self.strict_match_precision += 1
                    match_word = '-- strict match'

                if debug >= 2:
                    print(f' gold annotation: {g.annotation} {match_word}')

            else:
                s = system_timex[tid]
                results.confusion_matrix['UNDEFINED'][s.type] += 1
                if debug >= 1.5:
                    print(f' system annotation not found in gold: {system_timex[tid].annotation}')

        self.system_timex_count = len(system_timex)
        if self.system_timex_count != 0:
            strict_timex_precision = self.strict_match_precision / self.system_timex_count
            relaxed_timex_precision = self.relaxed_match_precision / self.system_timex_count
        else:
            strict_timex_precision = 0
            relaxed_timex_precision = 0

        if debug >= 1:
            print('\nTIMEX EXTRACTION PRECISION PERFORMANCE')
            print(f'Strict Precision: {strict_timex_precision:.6f}')
            print(f'Relaxed Precision: {relaxed_timex_precision:.6f}')

        if debug >= 2:
            print(f'total: {self.system_timex_count}')
            print(f'strict count: {self.strict_match_precision}')
            print(f'relaxed count: {self.relaxed_match_precision}')


@dataclass
class TimexClass:
    """
    Defining class for timex annotations
    """
    tid: str = ''
    text: str = ''
    type: str = ''
    value: str = ''
    annotation: str = ''


def extract_name(filename):
    parts = re.split('/', filename)
    length = len(parts)
    return parts[length-1]


def get_text(file_text):
    """
    Get only the annotated texts inside of TEXT tags 
    :param file_text: Content of a file in XML form.
    :return: extracted TEXT content, and original input text
    """
    import xml.dom.minidom
    dom = xml.dom.minidom.parseString(file_text.encode("utf-8"))
    xmlTag = dom.getElementsByTagName('TEXT')[0].toxml()
    xmlData = xmlTag.replace('<TEXT>\n', '').replace('</TEXT>', '')
    xmlData = re.sub('&quot;', '"', xmlData)
    return xmlData, file_text


def read_file(filename):
    with open(filename, "r", encoding="utf-8") as fileObj:
        file_text = fileObj.read()
    file_text = re.sub('&quot;', '"', file_text)

    return file_text


def get_entity_value(line, entity):
    """
    Given the event annotation, get the value for the features,
    e.g. <EVENT class=\"OCCURRENCE\" eid=\"e3\"> would return e3 for eid
    """
    if re.search(entity+'="[^"]*"', line):
        entity_text = re.findall(entity+'="[^"]*"', line)[0]
        entity_text = re.sub(entity+'=', '', entity_text).strip('"')
    else:
        entity_text = entity
    return entity_text


def strip_text(text):
    text = re.sub('<[^>]*>', '', text)
    return text


def get_attributes(tml_file):
    timex_attributes = {}

    all_timex = re.findall('<TIMEX3[^>]*>[^<]*</TIMEX3>', tml_file)
    for instance in all_timex:
        y = TimexClass()
        tid = get_entity_value(instance, 'tid')
        if tid == 't0':
            continue
        y.tid = tid
        if re.search('type=', instance):
            y.type = get_entity_value(instance, 'type')
        if re.search('value=', instance):
            y.value = get_entity_value(instance, 'value')
        y.text = re.sub('<[^>]*>', '', instance)
        y.annotation = instance
        timex_attributes[tid] = y

    return timex_attributes


def evaluate_two_files(gold_file, system_file, results):
    """
    Evaluate two files.
    """
    if debug >= 1:
        print('\n\nEVALUATE:', system_file, 'AGAINST GOLD ANNOTATION:', gold_file)

    gold_only_text, gold_tml = get_text(read_file(gold_file))
    system_only_text, system_tml = get_text(read_file(system_file))

    if strip_text(gold_only_text) != strip_text(system_only_text):
        print('TEXTS NOT SAME for', system_file, 'against', gold_file)
        print('EXITING THE EVALUATION SCRIPT')
        gold_text_lines = strip_text(gold_only_text).split('\n')
        system_text_lines = strip_text(system_only_text).split('\n')
        for i in range(0, len(gold_text_lines)):
            if i < len(gold_text_lines) and i < len(system_text_lines):
                if gold_text_lines[i] != system_text_lines[i]:
                    print('gold: "' + gold_text_lines[i] + '"')
                    print('syst: "' + system_text_lines[i] + '"')
                    print('')
            else:
                print('system missing gold\'s line number', i)
                print('gold: "' + gold_text_lines[i] + '"')
                print('')

        if len(system_text_lines) > len(gold_text_lines):
            for i in range(0, len(system_text_lines)):
                if i >= len(gold_text_lines):
                    print('gold doesn\'t have system\'s line number', i)
                    print('syst: "' + system_text_lines[i] + '"')
                    print('')

        sys.exit(1)

    gold_timex = get_attributes(gold_tml)
    system_timex = get_attributes(system_tml)

    current_results = TimexResults()
    current_results.update_results(gold_timex, system_timex)

    # Adds individual attributes to parent result object
    results += current_results


def evaluate_two_folders(gold, system, results):
    if gold[-1] != '/':
        gold += '/'
    if system[-1] != '/':
        system += '/'
    for file in os.listdir(gold):
        if os.path.isdir(gold+file):
            subdir = file + '/'
            if debug >= 1:
                print('Traverse files in Directory', gold+subdir)
            evaluate_two_folders(gold+subdir, system+subdir, results)
        else:
            gold_file = gold + file
            system_file = system + file
            if not re.search('DS_Store', file):
                if debug >= 3:
                    print(gold_file, system_file)
                evaluate_two_files(gold_file, system_file, results)


def get_fscore(p, r):
    if p + r == 0:
        return 0
    return 2.0 * p * r / (p + r)


def get_performance(results):
    if debug >= 0.5:
        print("\n\n===============================================")
        print('\nDETAIL OVERALL PERFORMANCES')

    if results.gold_timex_count != 0:
        strict_timex_recall = results.strict_match_recall / results.gold_timex_count
        relaxed_timex_recall = results.relaxed_match_recall / results.gold_timex_count
    else:
        strict_timex_recall = 0
        relaxed_timex_recall = 0

    if debug >= 0.5:
        print('\nTIMEX EXTRACTION RECALL PERFORMANCE')
        print(f'Strict Recall: {strict_timex_recall:.6f}')
        print(f'Relaxed Recall: {relaxed_timex_recall:.6f}')

    if results.system_timex_count != 0:
        strict_timex_precision = results.strict_match_precision / results.system_timex_count
        relaxed_timex_precision = results.relaxed_match_precision / results.system_timex_count
    else:
        strict_timex_precision = 0
        relaxed_timex_precision = 0

    if debug >= 0.5:
        print('\nTIMEX EXTRACTION PRECISION PERFORMANCE')
        print(f'Strict Precision: {strict_timex_precision:.6f}')
        print(f'Relaxed Precision: {relaxed_timex_precision:.6f}')

    if debug >= 0.5:
        print('\nTIMEX FEATURE EXTRACTION PERFORMANCE')
        print(f'total gold timex or total features in gold data: {results.gold_timex_count}')
        print(f'total gold timex or total features in system data: {results.system_timex_count}')
        print(f'matching timex: {results.relaxed_match_recall}')
        if results.relaxed_match_recall != 0:
            print(f'matching type: {results.type_match} '
                  f'accuracy: {results.type_match / results.relaxed_match_recall:.6f} '
                  f'precision: {results.type_match / results.relaxed_match_recall * relaxed_timex_precision:.6f} '
                  f'recall: {results.type_match / results.relaxed_match_recall * relaxed_timex_recall:.6f}')
            print(f'matching value: {results.value_match} '
                  f'accuracy: {results.value_match / results.relaxed_match_recall:.6f} '
                  f'precision: {results.value_match / results.relaxed_match_recall * relaxed_timex_precision:.6f} '
                  f'recall: {results.value_match / results.relaxed_match_recall * relaxed_timex_recall:.6f}')
        else:
            print('timex attribute accuracy: 0')
        print('\n')

    if results.system_timex_count == 0 and results.gold_timex_count == 0:
        strict_timex_fscore = 1
        relaxed_timex_fscore = 1
        performance_type = 1
        performance_value = 1
    else:
        strict_timex_fscore = get_fscore(strict_timex_precision, strict_timex_recall)
        relaxed_timex_fscore = get_fscore(relaxed_timex_precision, relaxed_timex_recall)

        if results.relaxed_match_recall != 0:
            performance_type = results.type_match / results.relaxed_match_recall * relaxed_timex_fscore
            performance_value = results.value_match / results.relaxed_match_recall * relaxed_timex_fscore
        else:
            performance_type = 0
            performance_value = 0

    if debug >= 0:
        print('=== Timex Performance ===')
        print('Strict Match\tF1\tP\tR')
        print(f'\t\t{strict_timex_fscore * 100:.2f}\t'
              f'{strict_timex_precision * 100:.2f}\t'
              f'{strict_timex_recall * 100:.2f}')

        print('Relaxed Match\tF1\tP\tR')
        print(f'\t\t{relaxed_timex_fscore * 100:.2f}\t'
              f'{relaxed_timex_precision*100:.2f}\t'
              f'{relaxed_timex_recall * 100:.2f}')

        print('Attribute F1\tValue\tType')
        print(f'\t\t{performance_value * 100:.2f}\t{performance_type * 100:.2f}\n')

        print(f'Machine-Readable results:\t'
              f'{strict_timex_fscore * 100:.2f}\t'
              f'{strict_timex_precision * 100:.2f}\t'
              f'{strict_timex_recall * 100:.2f}\t'
              f'{relaxed_timex_fscore * 100:.2f}\t'
              f'{relaxed_timex_precision*100:.2f}\t'
              f'{relaxed_timex_recall * 100:.2f}\t'
              f'{performance_value * 100:.2f}\t'
              f'{performance_type * 100:.2f}')


if __name__ == '__main__':

    debug = float(get_arg(3))
    invalid = False
    gold_arg = get_arg(1)
    system_arg = get_arg(2)
    if len(sys.argv) < 3:
        invalid = True
    results = TimexResults()
    if invalid:
        raise ValueError('Invalid input format.\n'
                         'To check the performance of a single file:\n'
                        '\tpython evaluate_entities.py gold_file_path system_file_path\n'
                         'To check the performance of all files in a gold folder:\n'
                         '\t python evaluate_entities.py gold_folder_path system_folder_path \n\n')
    # both arguments are directories
    elif os.path.isdir(gold_arg) and os.path.isdir(system_arg):
        # for each file in the gold folder, check the performance of that file in the system folder
        if debug >= 2:
            print('compare files in two folders')
        evaluate_two_folders(gold_arg, system_arg, results)
    # both arguments are files
    elif os.path.isfile(gold_arg) and os.path.isfile(system_arg):
        # compare the performance between two files
        if debug >= 2:
            print('compare two files')
        evaluate_two_files(gold_arg, system_arg, results)
    else:
        raise ValueError('Mismatch between data format of gold and system files.\n'
                         'Supply either only single files or two folders.')

    get_performance(results)
    if debug >= 1:
        print("===================================")
        print("\nTYPE CONFUSION MATRIX\n")
        print(results.confusion_matrix)
