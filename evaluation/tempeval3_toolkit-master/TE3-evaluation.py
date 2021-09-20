#!/usr/bin/python 

# this program evaluates the performance of extracted events and temporal expressions,
# and the overall temporal relation performance

# $ cd tools

# > python TE3-evaluation.py gold_folder_or_file system_folder_or_filefile
# $ python TE3-evaluation.py data/gold data/system
# # runs with debug level 0 and only reports the performance; also creates a temporary folder to create normalized files

# > python TE3-evaluation.py gold_folder_or_file system_folder_or_filefile debug_level
# $ python TE3-evaluation.py data/gold data/system 1
# # based on the debug_level print debug information

# > python TE3-evaluation.py gold_folder_or_file system_folder_or_filefile debug_level tmp_folder
# $ python TE3-evaluation.py data/gold data/system 1 tmp_folder
# # additionally creates the temporary folder to put normalized files, which could be used for later uses


# usage:
# to check the performance of a single file:
#    python TE3-evaluation.py gold_file_path system_file_path debug_level
# to check the performace of all files in a gold folder:
#    python TE3-evaluation.py gold_folder_path system_folder_path debug_level


# V 1.0 Naushad UzZaman, March 24, 2012
# Python3 port: Dennis Aumiller and Satya Almasian, July 2021

import os
import re
import sys
import tempfile


def get_arg(index):
    return sys.argv[index]


def get_directory_path(path):
    name = extract_name(path)
    directory = re.sub(name, '', path)
    if directory == '':
        directory = './'
    return directory


def extract_name(filename):
    parts = re.split('/', filename)
    length = len(parts)
    return parts[length - 1]


def create_tmp_folder():
    # create temporary folder
    if os.path.exists(directory_path + 'tmp-to-be-deleted'):
        tmp_folder_command = f'rm -rf {directory_path}tmp-to-be-deleted/*'
        os.system(tmp_folder_command)


def copy_folders():
    global gold_dir
    global system_dir

    gold_folder = sys.argv[1]
    system_folder = sys.argv[2]
    print(f'Machine-Readable path: {system_folder}')

    if len(sys.argv) <= 4:
        tmp_copy_folder = tempfile.mkdtemp()
    elif len(sys.argv) > 4:
        tmp_copy_folder = sys.argv[4]
        if tmp_copy_folder[-1] == '/':
            tmp_copy_folder = tmp_copy_folder[:-1]
        tmp_folder_command = f'mkdir {tmp_copy_folder}'
        try:
            os.system(tmp_folder_command)
            tmp_folder_command = f'mkdir {tmp_copy_folder}/gold'
            os.system(tmp_folder_command)
            tmp_folder_command = f'mkdir {tmp_copy_folder}/system'
            os.system(tmp_folder_command)

        except:
            raise PermissionError(f'Cannot create folder {tmp_copy_folder}')

    if os.path.isdir(gold_folder) and os.path.isdir(system_folder):
        if gold_folder[-1] != '/':
            gold_folder += '/'
        if system_folder[-1] != '/':
            system_folder += '/'

        try:
            tmp_folder_command = f'cp -r {gold_folder} {tmp_copy_folder}/gold/'
            os.system(tmp_folder_command)
            tmp_folder_command = f'cp -r {system_folder} {tmp_copy_folder}/system/'
            os.system(tmp_folder_command)
            gold_dir = f'{tmp_copy_folder}/gold/'
            system_dir = f'{tmp_copy_folder}/system/'
        except:
            raise PermissionError(f'Cannot copy to new folder')

    elif (not os.path.isdir(gold_folder)) and (not os.path.isdir(system_folder)):
        tmp_folder_command = f'cp {gold_folder} {tmp_copy_folder}/gold/'
        os.system(tmp_folder_command)
        tmp_folder_command = f'cp {system_folder} {tmp_copy_folder}/system/'
        os.system(tmp_folder_command)
        gold_dir = f'{tmp_copy_folder}/gold/'
        system_dir = f'{tmp_copy_folder}/system/'

    return tmp_copy_folder


def normalize_folders():
    global gold_dir
    global system_dir

    if debug >= 1:
        normalization_command = f'java -jar TimeML-Normalizer/TimeML-Normalizer.jar -d -a "{gold_dir};{system_dir}"'
    else:
        normalization_command = f'java -jar TimeML-Normalizer/TimeML-Normalizer.jar -a "{gold_dir};{system_dir}"'
    os.system(normalization_command)


def evaluate(tmp_folder):
    if len(os.listdir(gold_dir)) != len(os.listdir(sys.argv[1])):
        raise ValueError('Invalid TimeML XML file exists, NOT EVALUATING FILES')

    eval_command = f'python3 evaluation-entities/evaluate_entities.py ' \
                   f'{tmp_folder}/gold-normalized/ ' \
                   f'{tmp_folder}/system-normalized/ ' \
                   f'{debug}'
    os.system(eval_command)


if __name__ == '__main__':
    gold_dir = ''
    system_dir = ''

    directory_path = get_directory_path(get_arg(0))
    if len(sys.argv) > 3:
        debug = float(sys.argv[3])
    else:
        debug = 0

    create_tmp_folder()
    if debug >= 3:
        print('folder created')
    tmp_folder = copy_folders()
    if debug >= 3:
        print('copy folder')
    normalize_folders()
    if debug >= 3:
        print('normalized')
    evaluate(tmp_folder)
    if debug >= 3:
        print('evaluated')

    if len(sys.argv) <= 4:
        command = 'rm -rf ' + tmp_folder
        if debug >= 1:
            print('Deleting temporary folder', tmp_folder)
            print('To keep the temporary folder, RUN: '
                  '"python TE3-evaluation.py gold_folder_or_file system_folder_or_filefile debug_level tmp_folder"')
        os.system(command)