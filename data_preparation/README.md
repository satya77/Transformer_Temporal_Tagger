# Data preparation scripts

This folder contains code for data preparation for all the models. The seq2seq format is different than the BIO tags. Moreover,
the script `gather_corpus_stats.py`, creates the statistics table from the paper and gathers the tag information from the different data sources.

### Heideltime
The subfolder `heideltime` contains the code related to the HeidelTime package. To run the code you need the [heideltime python wrapper](https://github.com/PhilipEHausner/python_heideltime/tree/master/python_heideltime)
installed. `heideltime_generate_tempeval_data.py` tags the text from the benchmark data using the heideltime python package.
We add the TimeML tags to the beginning and end of the documents and try to match the formatting of the sources.
We used this script to compute the type F1 for Wikiwars and Tweets and also to compute the class confusion matrices.
An example of the usages is below:
```bash
python heideltime_generate_tempeval_data.py --input_folder ./data/temporal/tempeval/tempeval_test \
--output_folder ./results/baselines/heideltime/tempeval --tweets False 
```
Where the input files are located in the path `input_folder` and the processed files will be stored in `output_folder`.
The `tweets` tag should be set if the input data is the Tweets dataset since it has a different format than Wikiwars and TempEval data.
`subset_heideltime_data.py` subsets the full weakly labeled heideltime data, to keep only 1 million instances, from which
roughly 26 percent are negative examples and contain no annotations.

### Seq2seq data
To generate examples for seq2seq models use `seq2seq_data_generator.py` as follows:

```bash
python seq2seq_data_generator.py --input-dir-train ../data/temporal/tempeval/tempeval_train/TimeBank,../data/temporal/tempeval/tempeval_train/AQUAINT,../data/temporal/wikiwars/trainingset/tml,../data/temporal/tweets/trainingset/tml \
--input-dir-test ../data/temporal/tempeval/tempeval_test \
--output-file-train ./data/seq2seq/train/train_mixed.json \
--output-file-test ../data/seq2seq/test/tempeval_test.json 
```
Test and train data folders are specified using `input-dir-train` and `input-dir-test`.
The train data can take several folders, to generate a mixed training dataset. The folder paths should be separated by commas.
The output file paths for train and test are specified in `output-file-train` and `output-file-test`.
They should both have `.json` data type.
The script will go over each file in the input directories and extract the timex3 tags and the raw text.
The processed documents are placed in a dictionary with three attributes. `text` to mark the raw text.
`tagged_text` is the text containing the timex3 tag and `date` indicating the reference date of the document.
Additionally, the script prints out some statistics about each data source and the number of tags and paragraphs.

###BIO tagged data
For token classifiers, we need to convert the timex3 tags and their types into an extended BIO tags schema. We define
9 classes to cover all types:
```
O -- outside of a tag 
I-TIME -- inside tag of time 
B-TIME -- beginning tag of time
I-DATE -- inside tag of date 
B-DATE -- beginning tag of date
I-DURATION -- inside tag of duration 
B-DURATION -- beginning tag of duration
I-SET -- inside tag of the set 
B-SET -- beginning tag of the set
```
For easier processing, we input the `.json` files from seq2seq data. Use the `preprocess_BIO_tags.py` script as follows:

```bash
python preprocess_BIO_tags.py --input-file-train ../data/seq2seq/train/tempeval_train.json \
--input-file-test ../data/seq2seq/test/tempeval_test.json \
--output-file-train ../data/BIO/train_staging/tempeval_train.txt \
--output-file-test ../data/BIO/test_staging/tempeval_test.txt 
```
Where test and train files are specified using `input-file-train` and `input-file-test`.
The output file paths for train and test are specified in `output-file-train` and `output-file-test` and both have `.txt` format.
The script looks at the `tagged_text` column and translates the tags into BIO scheme.
Each file is separated by a commented line that has the reference date of the document.
For the files to be used by the hugginface models, additional preprocessing is needed.
First We will only keep the word column and the tag column for our train, dev, and test datasets.
Therefore, for all dataset you need to run:
``` bash
cat ../data/BIO/train_staging/tempeval_train.txt | cut -d " " -f 1,2 > ../data/BIO/train_staging/tempeval_train_temp.txt
```
Moreover, you need a file with distinct list of the tags available:
```bash 
cat ../data/BIO/train_staging/tempeval_train.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > ../data/BIO/train_staging/labels.txt

```
The script `preprocess_BIO_tags.py`  will split sentences longer than MAX_LENGTH (in terms of tokens) into small ones. Otherwise, long sentences will be truncated when tokenized, causing the loss of training data and some tokens in the test set not being predicted.
An example of the usage is seen below:

``` python
python preprocess_BIO_tags.py ../data/BIO/train_staging/tempeval_train_temp.txt "bert-base-uncased" 512 > ../data/BIO/normal/tempeval_train.txt
```
The out of the `preprocess_BIO_tags.py` script can directly be used for training with a list of labels (`labels.txt`).
The staging files and output of each stage are available in the data folder. 
