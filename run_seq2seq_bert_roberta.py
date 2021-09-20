from argparse import ArgumentParser

from transformers import RobertaTokenizerFast, BertTokenizerFast
from datasets import load_dataset
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from transformers import EncoderDecoderModel
from temporal_models import seq2seq_utils
import os

from temporal_models.seq2seq_utils import DataProcessor


##########------------------------------- Parameters -------------------------------##########

def get_args():
    args = ArgumentParser()
    args.add_argument("--transformer_cache", type=str, default="./cache/",
                      help="Folder for hugginface cache.")
    args.add_argument("--data_set_cache", type=str, default="./cache/",
                      help="Folder for dataset cache.")
    args.add_argument("--pre_train", type=bool, default=False,
                      help="Should you pretrain or fine_tune. ")
    args.add_argument("--model_dir", type=str, required=True,
                      help="folder to save the moel")
    args.add_argument("--pretrain_path", type=str,
                      help="the pretrained model to load incase of fine tunning.")
    args.add_argument("--model_name", type=str, required=True,
                      help="name of the model roberta-base or bert-base-uncased")
    args.add_argument("--train_data", type=str, default="./data/temporal/tempeval_seq2seq/train/train_mixed.json",
                      help="data for the train")

    args.add_argument("--eval_data", type=str, default="./data/temporal/tempeval_seq2seq/test/tempeval_test.json",
                      help="data for evaluation")
    args.add_argument("--text_column", type=str, default="text",
                      help="column of the data the has the text")
    args.add_argument("--tag_column", type=str, default="tagged_text",
                      help="column of the data the has the tagged text")
    args.add_argument("--date_column", type=str, default="date",
                      help="column of the data the has the date")

    args.add_argument("--max_length", type=int, default=512,
                      help="max input len")
    args.add_argument("--batch_size", type=int, default=12,
                      help="batch size on each gpu (if multiple available). ")
    args.add_argument("--min_length", type=int, default=56,
                      help="max input len")
    args.add_argument("--no_repeat_ngram_size", type=int, default=3,
                      help="n-grams of this size can occur once")
    args.add_argument("--early_stopping", type=bool, default=True,
                      help="stop beam search when num_beams sentences are finished per batch.")
    args.add_argument("--length_penalty", type=int, default=2,
                      help="Set to values < 1.0 in order to encourage the model to generate shorter sequences, to a value > 1.0 in order to encourage the model to produce longer sequences.")
    args.add_argument("--num_beams", type=int, default=2,
                      help="numbers of beams for beam search")
    args.add_argument("--num_gpu", type=str, default="0",
                      help="the number of the gpus or gpus that you want to use, e.g., '0' for one and '0,1' for multiple.")
    args.add_argument("--seed", type=int, default=0,
                      help="optional a seed. ")
    args.add_argument("--num_train_epochs", type=int, default=3,
                      help="number of training epochs. ")
    args.add_argument("--save_steps", type=int, default=100_000,
                      help="steps to save a checkpoint ")
    args.add_argument("--eval_steps", type=int, default=250_000,
                      help="steps to run evalution ")
    args.add_argument("--warmup_steps", type=int, default=1000,
                      help="The number of steps for the warmup phase.")

    return args.parse_args()
##########------------------------------- Main -------------------------------##########

if __name__ == "__main__":
    args=get_args()
    os.environ['TOKENIZERS_PARALLELISM'] = "false"
    os.environ['TRANSFORMERS_CACHE'] = args.transformer_cache
    os.environ['DATASETS_CACHE'] = args.data_set_cache
    os.environ["CUDA_VISIBLE_DEVICES"]= args.num_gpu

    ##########------------------------------- Get model -------------------------------##########

    if args.model_name.startswith("roberta"):
        tokenizer = RobertaTokenizerFast.from_pretrained(args.model_name)
    else:
        tokenizer = BertTokenizerFast.from_pretrained(args.model_name)

    tokenizer.bos_token = tokenizer.cls_token
    tokenizer.eos_token = tokenizer.sep_token

    if args.pre_train:

        model2model = EncoderDecoderModel.from_encoder_decoder_pretrained(args.model_name, args.model_name)
    else:
        if args.pretrain_path== None:
            print(" you need to specifiy a path to the pretrained model. ")
            exit()
        model2model = EncoderDecoderModel.from_pretrained(args.pretrain_path)

    # set special tokens
    model2model.config.decoder_start_token_id = tokenizer.bos_token_id
    model2model.config.eos_token_id = tokenizer.eos_token_id
    model2model.config.pad_token_id = tokenizer.pad_token_id

    # sensible parameters for beam search
    model2model.config.vocab_size = model2model.config.decoder.vocab_size
    model2model.config.max_length = args.max_length
    model2model.config.min_length = args.min_length
    model2model.config.no_repeat_ngram_size = args.no_repeat_ngram_size
    model2model.config.early_stopping = args.early_stopping
    model2model.config.length_penalty = args.length_penalty
    model2model.config.num_beams = args.num_beams


    ##########------------------------------- Process data -------------------------------##########

    data_processor= DataProcessor(tokenizer, args.text_column,args.tag_column,None,args.max_length,args.max_length,"max_length",args.date_column)
    process_data_to_model_inputs= data_processor.process_data_to_model_inputs_encoder_decoder

    data_files = {}


    data_files["train"] =args.train_data
    data_files["eval"] =args.eval_data
    folder_name = args.model_dir
    encoder_max_length = args.max_length
    decoder_max_length = args.max_length

    datasets = load_dataset("json", data_files=data_files,  cache_dir=args.data_set_cache)
    train_data = datasets["train"]
    val_data = datasets["eval"]


    train_data = train_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=args.batch_size,
    )
    train_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    val_data = val_data.map(
        process_data_to_model_inputs,
        batched=True,
        batch_size=args.batch_size,
    )
    val_data.set_format(
        type="torch", columns=["input_ids", "attention_mask", "decoder_input_ids", "decoder_attention_mask", "labels"],
    )

    ##########------------------------------- Training -------------------------------##########

    metrics = seq2seq_utils.metrics(tokenizer)

    # set training arguments
    training_args = Seq2SeqTrainingArguments(
        output_dir=args.model_dir,
        evaluation_strategy="steps",
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        predict_with_generate=True,
        logging_steps=10000,
        num_train_epochs=args.num_train_epochs,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        warmup_steps=args.warmup_steps,
        overwrite_output_dir=True,
        save_total_limit=3,
        seed = args.seed

    )
    # instantiate trainer
    trainer = Seq2SeqTrainer(
        model=model2model,
        tokenizer=tokenizer,
        args=training_args,
        compute_metrics=metrics.compute_metrics,
        train_dataset=train_data,
        eval_dataset=val_data,
    )

    train_result=trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    trainer.save_model(args.model_dir)