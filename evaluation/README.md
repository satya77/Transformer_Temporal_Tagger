# Evaluation script and code to generate the relevant data
The folder `tempeval3_toolkit-master` contains the evaluation toolkit from the tempeval challenge.
We only added a few lines to the main file to compute the confusion matrices if the debug level is set to above 1 and also
to print out the result in a machine-readable format.

### Transforming classification output to tags
Use `classifier_generate_tempeval_data.py`, an example of the usage:
```
python classifier_generate_tempeval_data.py --input_dir  ./data/temporal/wikiwars/wikiwars_test_with_newline/  \
  --output_dir ./results/token_clasification/fine_tune_wikiwars/wikiwars_test_crf_bert_no_pretrain_8epochs_seed_19 \
  --model_dir ./fine_tune_wikiwars/bert_crf_tagging_no_pretrain_8epochs/bert_crf_tagging_seed_19 \
  --model_type crf 
```
The script runs the token classifier on the files given in `input_dir` and looks for any tags starting with `B` or `I`,
indicating a presence of temporal information. It will perform majority voting on the tags of the wordpieces and decides on the final label of a word. The word is then identified in the original text and the Timex tag is created with
the respective type. The new texts are placed in `output_dir`.

`model_dir` must contain the path to the pre-trained model and `model_type` defines the pre-trained model type from `normal`,
`date` and `crf`.

### Transforming seq2seq output to tags
Use `seq2seq_generate_tempeval_data.py`, an example of the usage:
```
python seq2seq_generate_tempeval_data.py --input_dir  ./data/temporal/tempeval/tempeval_test  \
  --output_dir ./results/seq2seq/tempeval/fine_tune_mixed/tempeval_test_seq2seq_roberta_67 \
  --model_path ./fine_tune/roberta2roberta_fine_tuned_no_prefix/roberta2roberta_fine_tune_no_prefixed_seed_67 \
  --dataset_type tempeval \
  --model_type roberta  
```
The script loads the model from `model_path` and uses the tokenizer from the `model_type` to tokenize and prepare the documents in `input_dir`. Each document is divided into paragraphs and all the paragraphs are made into a batch that is fed into the model for prediction. The generated paragraphs are then cleaned using extensive rules and regex and read as an XML file. The text of the tags is matched against the original input and the tags that are found are replaced in the text.
If the full text does not match, the script tries to split the multi-words to perform partial matching.
If the unmatched word is a single word, we look at prefix matching.
If nothing is matched, the tag is ignored.
The final output is stored in `output_dir`.
You should set the `dataset_type` for the specific format of each dataset to be considered. 
