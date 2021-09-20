# Utilities for temporal models
Contains code for the two variants of the token classification (With extra date embedding and CRF layer) as well as
utility functions for training the seq2seq models in `seq2seq_utils.py`.
## BERT with CRF
The model definition and CRF layer are in `BERTWithCRF`, the code is adapted from [here](https://github.com/Louis-udm/NER-BERT-CRF/blob/master/NER_BERT_CRF.py).
To run this model with huggingface example code we need to define additional data processing classes in `ner_tasks_crf.py`.
## BERT with Extra Date Layer

The model definition is in `BERTWithDateLayerTokenClassification.py` for the additional date embedding we define
our vocabulary and tokenizer, which is defined in `NumBertTokenizer.py`. Respective feature generation code
is located in `ner_tasks_date_extra_layer.py` and we added `TokenClassificationWithDate` class to `ner_utils.py`.
The extra pre-processing also takes the comment line in the BIO tag data, which indicates the reference date into account.
The reference date is fed into the date embedding layer separately and is encoded using `NumBertTokenizer`.
The date embedding layer follows the format of wordpiece embeddings of BERT and its output is concatenated to each wordpiece embedding. 
