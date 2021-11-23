# BERT Got a Date: Introducing Transformers to Temporal Tagging
Satya Almasian*, Dennis Aumiller*, and Michael Gertz  
Heidelberg University  
Contact us via: `<lastname>@informatik.uni-heidelberg.de`

Code and data for the paper [BERT Got a Date: Introducing Transformers to Temporal Tagging](https://arxiv.org/abs/2109.14927).

[**Check out our models on Huggingface!**](https://huggingface.co/satyaalmasian)

-----------------------------------------------

Temporal tagging is the task of identification of temporal mentions in text; these expressions can be further divided into different type categories, which is what we refer to as expression (type) classification.
This repository describes two different types of transformer-based temporal taggers, which are both additionally capable of expression classification.
We follow the TIMEX3 schema definitions in their styling and expression classes (notably, the latter are one of `TIME, DATE, SET, DURATION`). The available data sources for temporal tagging are in the TimeML format, which is essentially a form of XML with tags encapsulating temporal expressions.  
An example can be seen below:
```
Due to lockdown restrictions, <TIMEX3 tid="t1" type="DATE" value="2020">2020</TIMEX3> 
might go down as the worst economic <TIMEX3 tid="t2" type="DURATION" value="P1Y">year</TIMEX3>
in over <TIMEX3 tid="t3" type="DURATION" value="P1DE">a decade</TIMEX3>.
```
For more sample instances, look at the content of `data.zip`. Refer to the README file in the respective unzipped folder for more information.  


## Installation
You can now install the underlying models by simply running
```bash
python3 -m pip install .
```
after cloning this repository; this will automatically install all necessary dependencies. We're working on making the installation even easier by providing a package on PyPI, stay tuned for more!

This repository contains code for data preparation and training of a seq2seq model (encoder-decoder architectured initialized from encoder-only architectures, specifically BERT or RoBERTa), as well as three token classification encoders (BERT-based).  
The output of the models discussed in the paper is in the `results` folder. Refer to the README file in the folder for more information.

**The zip files containing data & results are uploaded using Git LFS and require it as an additional library to work properly.**

To install Git LFS on Ubuntu: 

- Download/Install Git LFS (`sudo apt-get install git-lfs`, or download from https://git-lfs.github.com/)
- Run `git lfs install` 

If you want to generate data with Heideltime, you will additionally have to set up [`python_heideltime`](https://github.com/PhilipEHausner/python_heideltime) as a wrapper.
Due to the project nature of Heideltime, this installation has to be performed manually.


## Data Preparation
The scripts to generate training data is in the subfolder [data_preparation](./data_preparation/).
For more usage information, refer to the README file in the subfolder.
The data used for training and evaluation are provided in zipped form in `data.zip`.


## Evaluation
For evaluation, we use a slightly modified version of the TempEval-3 evaluation toolkit ([original source here](https://github.com/naushadzaman/tempeval3_toolkit)).
We refactored the code to be compatible with Python3, and incorporated additional evaluation metrics, such as a confusion matrix for type classification.
We cross-referenced results to ensure full backward-compatibility and all runs result in the exact same results for both versions.
Our adjusted code, as well as scripts to convert the output of transformer-based tagging models are in the [evaluation](temporal_taggers/evaluation/) subfolder.
For more usage information, refer to the README file in the respective subfolder.


## Temporal models
We train and evaluate two types of setups for joint temporal tagging and classification:
* **Token Classification:** We define three variants of simple token classifiers; all of them are based on Huggingface's [BertForTokenClassification](https://huggingface.co/transformers/model_doc/bert.html#bertfortokenclassification). We adapt their "[token classification for named entity recognition script](https://github.com/huggingface/transformers/tree/master/examples/pytorch/token-classification)" to train these models. All the models are trained using `bert-base-uncased` as their pre-trained checkpoint.
* **Text-to-Text Generation (Seq2Seq):** These models are encoder-decoder architectures using BERT or RoBERTa for initial weights. We use Huggingface's [EncoderDecoder](https://huggingface.co/transformers/model_doc/encoderdecoder.html) class for initialization of weights, starting from `bert-base-uncased` and `roberta-base`, respectively.

### Seq2seq
To train the seq2seq models, use `run_seq2seq_bert_roberta.py`. Example usage is as follows:
```bash
python3 run_seq2seq_bert_roberta.py --model_name roberta-base --pre_train True \
--model_dir ./test --train_data ./data/seq2seq/train/tempeval_train.json \ 
--eval_data ./data/seq2seq/test/tempeval_test.json --num_gpu 2 --num_train_epochs 1 \
warmup_steps 100 --seed 0 --eval_steps 200
```
Which trains a roberta2roberta model defined by `model_name` for `num_train_epochs` epochs on the gpu with ID `num_gpu`.
The random seed is set by `seed` and the number of warmup steps by `warmup_steps`.
Train data should be specified in `train_data` and `model_dir` defines where the model is saved.
set `eval_data` if you want intermediate evaluation defined by `eval_steps`.
If the `pre_train` flag is set to true it will load the checkpoints from the Huggingface hub and fine-tune on the dataset given.
If the `pre_train` is false, we are in the fine-tuning mode, and you can provide the path to the pre-trained model with `pretrain_path`.
We used the `pre_train` mode to train on weakly labeled data provided by the rule-based system of HeidelTime and set the `pre_train` to false
for fine-tuning on the benchmark datasets. If you wish to simply fine-tune the benchmark datasets using the Huggingface checkpoints
you can set the `pre_train` to ture, as displayed in the example above.
For additional arguments such as length penalty, the number of beams, early stopping, and other model specifications, please refer to the script.

### Token Classifiers
As mentioned above all token classifiers are trained using an adaptation of the NER script from Huggingface. To train these models use
`run_token_classifier.py` like the following example:
```bash
python3 run_token_classifier.py --data_dir /data/temporal/BIO/wikiwars \ 
--labels ./data/temporal/BIO/train_staging/labels.txt \ 
--model_name_or_path bert-base-uncased \ 
--output_dir ./fine_tune_wikiwars/bert_tagging_with_date_no_pretrain_8epochs/bert_tagging_with_date_layer_seed_19 --max_seq_length  512  \
--num_train_epochs 8 --per_device_train_batch_size 34 --save_steps 3000 --logging_steps 300 --eval_steps 3000 \ 
--do_train --do_eval --overwrite_output_dir --seed 19 --model_date_extra_layer    
```
We used `bert-base-uncased ` as the base of all our token classification models for pre-training as defined by `model_name_or_path`.
For fine-tuning on the datasets `model_name_or_path` should point to the path of the pre-trained model. `labels` file is created during data preparation for more information refer to the [subfolder](./data_preparation/README.md).
`data_dir` points to a folder that contains `train.txt`, `test.txt` and `dev.txt` and `output_dir` points to the saving location.
You can define the number of epochs by `num_train_epochs`, set the seed with `seed` and batch size on each GPU with `per_device_train_batch_size`.
For more information on the parameters refer to the [Huggingface script](https://github.com/huggingface/transformers/tree/master/examples/pytorch/token-classification).
In our paper, we introduce 3 variants of token classification, which are defined by flags in the script.
If no flag is set the model trains the vanilla BERT for token classification.
The flag `model_date_extra_layer` trains the model with an extra date layer and `model_crf` adds the extra crf layer.
To train the extra date embedding you need to download the vocabulary file and specify its path in the `date_vocab` argument.
The description and model definition of the BERT variants are in folder [temporal_models](./temporal_models/).
Please refer to the README file for further information. For training different model types on the same data, make sure to remove
the cached dataset, since the feature generation is different for each model type. 

## Load directly from the Huggingface Model Hub
We uploaded our best-performing version of each architecture to the Huggingface Model Hub.
The weights for the other four seeding runs are available upon request.
We upload the variants that were fine-tuned on the concatenation of *all three* evaluation sets for better generalization to various domains.
Token classification models are variants without pre-training.
Both seq2seq models are pre-trained on the weakly labelled corpus and fine-tuned on the mixed data. 

Overall we upload the following five models. For other model configurations and checkpoints please get in contact with us:

* [satyaalmasian/temporal_tagger_roberta2roberta](https://huggingface.co/satyaalmasian/temporal_tagger_roberta2roberta): Our best perfoming model from the paper, an encoder-decoder architecture using RoBERTa.
  The model is pre-trained on weakly labeled news articles, tagged with HeidelTime, and fined-tuned on the train set of TempEval-3, Tweets, and Wikiwars.
* [satyaalmasian/temporal_tagger_bert2bert](https://huggingface.co/satyaalmasian/temporal_tagger_bert2bert): Our second seq2seq model , an encoder-decoder architecture using BERT.
  The model is pre-trained on weakly labeled news articles, tagged with HeidelTime, and fined-tuned on the train set of TempEval-3, Tweets, and Wikiwars.
* [satyaalmasian/temporal_tagger_BERT_tokenclassifier](https://huggingface.co/satyaalmasian/temporal_tagger_BERT_tokenclassifier): BERT for token classification model or vanilla BERT model from the paper.
  This model is only trained on the train set of TempEval-3, Tweets, and Wikiwars.
* [satyaalmasian/temporal_tagger_DATEBERT_tokenclassifier](https://huggingface.co/satyaalmasian/temporal_tagger_DATEBERT_tokenclassifier): BERT for token classification with an extra date embedding, that encodes the reference date of the
  document. If the document does not have a reference date, it is best to avoid this model. Moreover, since the architecture
  is a modification of a default Huggingface model, the usage is not as straightforward and requires the classes defined in the `temporal_model`
  module. This model is only trained on the train set of TempEval-3, Tweets, and Wikiwars.
* [satyaalmasian/temporal_tagger_BERTCRF_tokenclassifier](https://huggingface.co/satyaalmasian/temporal_tagger_BERTCRF_tokenclassifier) :BERT for token classification with a CRF layer on the output. Moreover, since the architecture
  is a modification of a default Huggingface model, the usage is not as straightforward and requires the classes defined in the `temporal_model`
  module. This model is only trained on the train set of TempEval-3, Tweets, and Wikiwars.

In the `examples` module, you find two scripts `model_hub_seq2seq_examples.py` and `model_hub_tokenclassifiers_examples.py` for
seq2seq and token classification examples using the Huggingface model hub. The examples load the models and use them on example sentences
for tagging. The seq2seq example uses the pre-defined post-processing from the tempeval evaluation and contains rules for the cases we came across in the benchmark dataset. 
If you plan to use these models on new data, it is best to observe the raw output of the first few samples to detect possible format problems that are easily fixable.
Further fine-tuning of the models is also possible.
For seq2seq models you can simply load the models with
```python3
tokenizer = AutoTokenizer.from_pretrained("satyaalmasian/temporal_tagger_roberta2roberta")
model = EncoderDecoderModel.from_pretrained("satyaalmasian/temporal_tagger_roberta2roberta")
```
and use the `DataProcessor` from `temporal_models.seq2seq_utils` to preprocess the `json` dataset. The model
can be fine-tuned using `Seq2SeqTrainer` (same as in `run_seq2seq_bert_roberta.py`).
For token classifiers the model and the tokenizers are loaded as follows:
```python3
tokenizer = AutoTokenizer.from_pretrained("satyaalmasian/temporal_tagger_BERT_tokenclassifier", use_fast=False)
model = BertForTokenClassification.from_pretrained("satyaalmasian/temporal_tagger_BERT_tokenclassifier")
```
Classifiers need a BIO-tagged file that can be loaded using `TokenClassificationDataset` and fine-tuned with the Huggingface `Trainer`.
For more information on the usage of these models refer to their model hub page. 


## Citation
If you use our models in your work, we would appreciate attribution with the following citation:
```
@article{almasian2021bert,
  title={{BERT got a Date: Introducing Transformers to Temporal Tagging}},
  author={Almasian, Satya and Aumiller, Dennis and Gertz, Michael},
  journal={arXiv preprint arXiv:2109.14927},
  url={https://arxiv.org/abs/2109.14927},
  year={2021}
}
```
