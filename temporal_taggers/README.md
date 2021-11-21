# Utilities for temporal models
Contains code for the two custom token classification variants (with extra date embedding or CRF layer), as well as
utility functions for training the seq2seq models in `seq2seq/utils.py`.

## BERT with CRF
The model definition and CRF layer are in `BERTWithCRF`, the code is adapted from [here](https://github.com/Louis-udm/NER-BERT-CRF/blob/master/NER_BERT_CRF.py).
To run this model with Huggingface, we need to define additional data processing classes in `ner_tasks_crf.py`.

## BERT with Extra Date Layer
The model definition is in `BERTWithDateLayerTokenClassification.py`.
For the additional date embedding we define a small custom vocabulary and tokenizer, which can be found in `NumBertTokenizer.py`. 
The corresponding feature generation code is located in `ner_tasks_date_extra_layer.py` and we added a `TokenClassificationWithDate` class to `ner_utils.py`.
The extra pre-processing also takes a comment line in the BIO tag data, which indicates the reference date.
This is important if you are planning t normalize temporal tags after the recognition.
The reference date is fed into the date embedding layer separately and is encoded using `NumBertTokenizer`.
The date embedding layer follows the format of the subword embeddings of BERT and its output is concatenated to each subword embedding of the regular text. 
