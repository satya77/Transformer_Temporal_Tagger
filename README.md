# BERT Got a Date: Introducing Transformers to Temporal Tagging
Code and data form the paper [BERT Got a Date: Introducing Transformers to Temporal Tagging] (arixlink).
The repository contains training code for two types of transformer-based temporal taggers capable of expression classification.
Temporal tagging is the task of identification of temporal information in text and classifying the expression into predefined types.
The temporal types are `TIME, DATE, SET, DURATION`. The available data sources for temporal tagging are in TMEML format, which an
XML file with TIMEX3 tags for the occurrence of any time and date information.
An example can be seen below:
```
The season started about a month earlier than usual, sparking concerns it might turn into the worst in <TIMEX3 tid="t2" type="DURATION" value="P1DE">a decade</TIMEX3>.
```
For more data instances, look at the `data` folder. Refer to the readme file in the folder for more information.

This repository contains code for data preparation and training of a seq2seq model (encoder-decoder architectured based on BERT or RoBERTa),
as well as, three token classifiers (BERT-based token classifier).

The output of the models discussed in the paper is in the `results` folder. Refer to the readme file in the folder for more information.

## Data Preparation

The scripts to generate training data is in subfolder [data_preparation](./data_preparation/README.md). For more usage information, refer to the readme file in the subfolder. The data used for training and evaluation are in `data` folder.


## Evaluation
For evaluation, we use the [tempeval3 toolkit](https://www.cs.york.ac.uk/semeval-2013/task1/index.php%3Fid=data.html).
We tweaked the code to add additional evaluation metrics, such as a confusion matrix. Their code, as well as, scripts to convert the output of transformer-based tagging models
are in subfolder [evaluation](./evaluation/README.md). For more usage information, refer to the readme file in the
subfolder.


## Temporal models
We train and evaluate two types of transformer architecture:
* **temporal tagging using token classification:** We define 3 token classifiers, for all of them we use [BERT for token classifcation](https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification)
  from huggingface as the base model. We adapt [token classification for named entity recognition script](https://github.com/huggingface/transformers/tree/master/examples/pytorch/token-classification)
  from the huggingface examples to train these models. All the models are trained using `bert-base-uncased` for the pre-trained checkpoint.
* **temporal tagging with text generation (seq2seq architecture):** Are seq2seq models are encoder-decoder architectures using
  BERT and RoBERTa. We use the [EncoderDecoder](https://huggingface.co/transformers/model_doc/encoderdecoder.html) model from hugginface for sequence generation tasks. We use the checkpoints `bert-base-uncased` and `roberta-base` for training.

## Seq2seq

To train the seq2seq models use `run_seq2seq_bert_roberta.py`. An example usage is as follows:
``` python
python run_seq2seq_bert_roberta.py --model_name roberta-base --pre_train True \
--model_dir ./test --train_data /export/data/salmasia/numbert/data/temporal/tempeval_seq2seq/train/tempeval_train.json \ 
--eval_data /export/data/salmasia/numbert/data/temporal/tempeval_seq2seq/test/tempeval_test.json --num_gpu 2 --num_train_epochs 1 \
warmup_steps 100 --seed 0 --eval_steps 200
```
Which trains a roberta2roberta model defined by `model_name` for `num_train_epochs` epochs on the gpu number `num_gpu`.
The random seed is set by `seed` and the number of warmup steps by `warmup_steps`.
Train data should be specified in `train_data` and `model_dir` defines where the model is saved.
set `eval_data` if you want intermediate evaluation defined by `eval_steps`.
If the `pre_train` flag is set to true it will load the checkpoints from the hugginface hub and fine-tune on the dataset given.
If the `pre_train` is false, we are in the fine-tuning mode and you can provide the path to the pre-trained model with `pretrain_path`.
We used the `pre_train` mode to train on weakly labeled data provided by the rule-based system of HeidelTime and set the `pre_train` to false
for fine-tunning on the benchmark datasets. If you wish to simply fine-tune the benchmark datasets using the huggingface checkpoints
you can set the `pre_train` to ture, as displayed in the example above.
For additional arguments such as length penalty, the number of beams, early stopping, and other model specifications, please refer to the script.

## Token Classifiers
As mentioned above all token classifiers are trained using an adaptation of the NER script from hugginface. To train these models use \
`run_token_classifier.py` like the following example:
```
python run_token_classifier.py --data_dir /data/temporal/BIO/wikiwars \ 
--labels ./data/temporal/BIO/train_staging/labels.txt \ 
--model_name_or_path bert-base-uncased \ 
--output_dir ./fine_tune_wikiwars/bert_tagging_with_date_no_pretrain_8epochs/bert_tagging_with_date_layer_seed_19 --max_seq_length  512  \
--num_train_epochs 8 --per_device_train_batch_size 34 --save_steps 3000 --logging_steps 300 --eval_steps 3000 \ 
--do_train --do_eval --overwrite_output_dir --seed 19 --model_date_extra_layer    
```
we used `bert-base-uncased ` as the base of all our models for pre-training as defined by `model_name_or_path`.
For fine-tuning on the datasets `model_name_or_path` should point to the path of the pre-trained model. `labels` file is created during data preparation for more information refer to the [subfolder](./data_preparation/README.md).
`data_dir` points to a folder that contains train.txt,test.txt and dev.txt and `output_dir` points to the saving location.
you can define the number of epochs by `num_train_epochs`, set the seed with `seed` and batch size on each GPU with `per_device_train_batch_size`. For more information on the parameters refer to [hugginface script](https://github.com/huggingface/transformers/tree/master/examples/pytorch/token-classification).
In our paper, we introduce 3 variants of token classification, which are defined by flags in the script.
If no flag is set the model trains the vanilla bert for token classification.
The flag `model_date_extra_layer` trains the model with an extra date layer and `model_crf` adds the extra crf layer.
To train the extra date embedding you need to download the vocabulary file and specify its path in `date_vocab` argument.
The description and model definition of the BERT variants are in folder [temporal_models](./temporal_models/README.md).
Please refer to its readme file for more information. For training different model types on the same data, make sure to remove
the cached dataset, since the feature generation is different for each model type. 

## Usage from the Model hub 
We uploaded one version of each model from our paper to huggingface model hub. Since we trained each model 5 times with different 
random seeds, we choosed the seed that had overall best performance. 

Although for token classifiers the best performing model for each benchmark dataset was trained soley on that data, we upload the models that are trained on the mixed data, since the objective is not 
do well on a single dataset but to be generalizable to more. The token classifier are without pretraining and trained on mixedd data. 

Both seq2seq models are pretrained on the weakly labled corpus and fine-tuned on the mixed data. 

Overall we upload the following 5 models, for other model configuration and checkpoints you can contact us:

* `satyaalmasian/temporal_tagger_roberta2roberta`
* `satyaalmasian/temporal_tagger_bert2bert`
* `satyaalmasi# BERT Got a Date: Introducing Transformers to Temporal Tagging
  Code and data form the paper [BERT Got a Date: Introducing Transformers to Temporal Tagging] (arixlink).
  The repository contains training code for two types of transformer-based temporal taggers capable of expression classification.
  Temporal tagging is the task of identification of temporal information in text and classifying the expression into predefined types.
  The temporal types are `TIME, DATE, SET, DURATION`. The available data sources for temporal tagging are in TMEML format, which an
  XML file with TIMEX3 tags for the occurrence of any time and date information.
  An example can be seen below:
```
The season started about a month earlier than usual, sparking concerns it might turn into the worst in <TIMEX3 tid="t2" type="DURATION" value="P1DE">a decade</TIMEX3>.
```
For more data instances, look at the `data` folder. Refer to the readme file in the folder for more information.

This repository contains code for data preparation and training of a seq2seq model (encoder-decoder architectured based on BERT or RoBERTa),
as well as, three token classifiers (BERT-based token classifier).

The output of the models discussed in the paper is in the `results` folder. Refer to the readme file in the folder for more information.

## Data Preparation

The scripts to generate training data is in subfolder [data_preparation](./data_preparation/README.md). For more usage information, refer to the readme file in the subfolder. The data used for training and evaluation are in `data` folder.


## Evaluation
For evaluation, we use the [tempeval3 toolkit](https://www.cs.york.ac.uk/semeval-2013/task1/index.php%3Fid=data.html).
We tweaked the code to add additional evaluation metrics, such as a confusion matrix. Their code, as well as, scripts to convert the output of transformer-based tagging models
are in subfolder [evaluation](./evaluation/README.md). For more usage information, refer to the readme file in the
subfolder.


## Temporal models
We train and evaluate two types of transformer architecture:
* **temporal tagging using token classification:** We define 3 token classifiers, for all of them we use [BERT for token classifcation](https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification)
  from huggingface as the base model. We adapt [token classification for named entity recognition script](https://github.com/huggingface/transformers/tree/master/examples/pytorch/token-classification)
  from the huggingface examples to train these models. All the models are trained using `bert-base-uncased` for the pre-trained checkpoint.
* **temporal tagging with text generation (seq2seq architecture):** Are seq2seq models are encoder-decoder architectures using
  BERT and RoBERTa. We use the [EncoderDecoder](https://huggingface.co/transformers/model_doc/encoderdecoder.html) model from hugginface for sequence generation tasks. We use the checkpoints `bert-base-uncased` and `roberta-base` for training.

### Seq2seq
To train the seq2seq models use `run_seq2seq_bert_roberta.py`. An example usage is as follows:
``` python
python run_seq2seq_bert_roberta.py --model_name roberta-base --pre_train True \
--model_dir ./test --train_data /export/data/salmasia/numbert/data/temporal/tempeval_seq2seq/train/tempeval_train.json \ 
--eval_data /export/data/salmasia/numbert/data/temporal/tempeval_seq2seq/test/tempeval_test.json --num_gpu 2 --num_train_epochs 1 \
warmup_steps 100 --seed 0 --eval_steps 200
```
Which trains a roberta2roberta model defined by `model_name` for `num_train_epochs` epochs on the gpu number `num_gpu`.
The random seed is set by `seed` and the number of warmup steps by `warmup_steps`.
Train data should be specified in `train_data` and `model_dir` defines where the model is saved.
set `eval_data` if you want intermediate evaluation defined by `eval_steps`.
If the `pre_train` flag is set to true it will load the checkpoints from the hugginface hub and fine-tune on the dataset given.
If the `pre_train` is false, we are in the fine-tuning mode and you can provide the path to the pre-trained model with `pretrain_path`.
We used the `pre_train` mode to train on weakly labeled data provided by the rule-based system of HeidelTime and set the `pre_train` to false
for fine-tunning on the benchmark datasets. If you wish to simply fine-tune the benchmark datasets using the huggingface checkpoints
you can set the `pre_train` to ture, as displayed in the example above.
For additional arguments such as length penalty, the number of beams, early stopping, and other model specifications, please refer to the script.

###Token Classifiers
As mentioned above all token classifiers are trained using an adaptation of the NER script from hugginface. To train these models use \
`run_token_classifier.py` like the following example:
```
python run_token_classifier.py --data_dir /data/temporal/BIO/wikiwars \ 
--labels ./data/temporal/BIO/train_staging/labels.txt \ 
--model_name_or_path bert-base-uncased \ 
--output_dir ./fine_tune_wikiwars/bert_tagging_with_date_no_pretrain_8epochs/bert_tagging_with_date_layer_seed_19 --max_seq_length  512  \
--num_train_epochs 8 --per_device_train_batch_size 34 --save_steps 3000 --logging_steps 300 --eval_steps 3000 \ 
--do_train --do_eval --overwrite_output_dir --seed 19 --model_date_extra_layer    
```
we used `bert-base-uncased ` as the base of all our models for pre-training as defined by `model_name_or_path`.
For fine-tuning on the datasets `model_name_or_path` should point to the path of the pre-trained model. `labels` file is created during data preparation for more information refer to the [subfolder](./data_preparation/README.md).
`data_dir` points to a folder that contains train.txt,test.txt and dev.txt and `output_dir` points to the saving location.
you can define the number of epochs by `num_train_epochs`, set the seed with `seed` and batch size on each GPU with `per_device_train_batch_size`. For more information on the parameters refer to [hugginface script](https://github.com/huggingface/transformers/tree/master/examples/pytorch/token-classification).
In our paper, we introduce 3 variants of token classification, which are defined by flags in the script.
If no flag is set the model trains the vanilla bert for token classification.
The flag `model_date_extra_layer` trains the model with an extra date layer and `model_crf` adds the extra crf layer.
To train the extra date embedding you need to download the vocabulary file and specify its path in `date_vocab` argument.
The description and model definition of the BERT variants are in folder [temporal_models](./temporal_models/README.md).
Please refer to its readme file for more information. For training different model types on the same data, make sure to remove
the cached dataset, since the feature generation is different for each model type.

## Pre-trained model and Model hub 
We uploaded one version of each model from our paper to huggingface model hub. Since we trained each model 5 times with different
random seeds, we chose the seed that had the overall best performance.

Although for token classifiers the best performing model for each benchmark dataset was trained solely on that data, we upload the models that are trained on the mixed data, since the objective is not to do well on a single dataset but to be generalizable to more. 
The token classifier modles are without pre-training on weakly labled data and fine-tunned on mixed training set of TempEval-3, Tweets and Wikiwars.

Both seq2seq models are pre-trained on the weakly labeled corpus and fine-tuned on the mixed data.

Overall we upload the following 5 models, for other model configurations and checkpoints you can contact us:

* [satyaalmasian/temporal_tagger_roberta2roberta](https://huggingface.co/satyaalmasian/temporal_tagger_roberta2roberta): Our best perfoming model from the paper, an encoder-decoder architecture using RoBERTa.
  The model is pre-trained on weakly labeled news articles, tagged with HeidelTime, and fined-tuned on the train set of TempEval-3, Tweets, and Wikiwars.
* [satyaalmasian/temporal_tagger_bert2bert](https://huggingface.co/satyaalmasian/temporal_tagger_bert2bert): Our second seq2seq model , an encoder-decoder architecture using BERT.
  The model is pre-trained on weakly labeled news articles, tagged with HeidelTime, and fined-tuned on the train set of TempEval-3, Tweets, and Wikiwars.
* [satyaalmasian/temporal_tagger_BERT_tokenclassifier](https://huggingface.co/satyaalmasian/temporal_tagger_BERT_tokenclassifier): BERT for token classification model or vanilla BERT model from the paper.
  This model is only trained on the train set of TempEval-3, Tweets, and Wikiwars.
* [satyaalmasian/temporal_tagger_DATEBERT_tokenclassifier](https://huggingface.co/satyaalmasian/temporal_tagger_DATEBERT_tokenclassifier): BERT for token classification with an extra date embedding, that encodes the reference date of the
  document. If the document does not have a reference date, it is best to avoid this model. Moreover, since the architecture
  is a modification of a default hugginface model, the usage is not as straightforward and requires the classes defined in the `temporal_model`
  module. This model is only trained on the train set of TempEval-3, Tweets, and Wikiwars.
* [satyaalmasian/temporal_tagger_BERTCRF_tokenclassifier](https://huggingface.co/satyaalmasian/temporal_tagger_BERTCRF_tokenclassifier) :BERT for token classification with a CRF layer on the output. Moreover, since the architecture
  is a modification of a default huggingface model, the usage is not as straightforward and requires the classes defined in the `temporal_model`
  module. This model is only trained on the train set of TempEval-3, Tweets, and Wikiwars.

In the `examples` module, you find two scripts `model_hub_seq2seq_examples.py` and `model_hub_tokenclassifiers_examples.py` for
seq2seq and token classification examples using the hugginface model hub. The examples load the models and use them on example sentences
for tagging. The seq2seq example uses the pre-defined post-processing from the tempeval evaluation and contains rules for the cases we came across in the benchmark dataset. 
If you plan to use these models on new data, it is best to observe the raw output of the first few samples to detect possible format problems that are easily fixable.
Further fine-tuning of the models is also possible.
For seq2seq models you can simply load the models as follows:
``` python
    tokenizer = AutoTokenizer.from_pretrained("satyaalmasian/temporal_tagger_roberta2roberta")
    model = EncoderDecoderModel.from_pretrained("satyaalmasian/temporal_tagger_roberta2roberta")
```
and use the `DataProcessor` from `temporal_models.seq2seq_utils` to preprocess the `json` dataset. The model
can be fine-tuned using `Seq2SeqTrainer` (same as `run_seq2seq_bert_roberta.py`).
For token classifiers the model and the tokenizers are loaded as follows:
``` python
    tokenizer = AutoTokenizer.from_pretrained("satyaalmasian/temporal_tagger_BERT_tokenclassifier", use_fast=False)
    model = BertForTokenClassification.from_pretrained("satyaalmasian/temporal_tagger_BERT_tokenclassifier")
```
classifiers need a BIO tagged file, that can be loaded using `TokenClassificationDataset` and fine-tuned with the hugginface `Trainer`.
For more information on the usage of these models refer to their model hub page. 


