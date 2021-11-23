"""
Script to generate timex3 files from the BIO tags of the classifiers, which can be used for tempeval evaluation script.
"""
import regex
import os
from argparse import ArgumentParser
from collections import Counter
from typing import List, Tuple
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, BertForTokenClassification
from bs4 import BeautifulSoup

from ..tagger import BERTWithDateLayerTokenClassification, BertWithCRF, DateTokenizer


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", default="../../data/temporal/tempeval/tempeval_test/",
                        help="Location of the files that should be annotated.")
    parser.add_argument("--output_dir", default="../../results/classifiers/fine_tune/tempeval_test_bert_crf_tagging_with_pretrain_seed_12",
                        help="Location where to write output files.")
    parser.add_argument("--file_ext", default="tml", help="File extension of files to be processed.")
    parser.add_argument("--model_dir", default="./model/fine_tune/bert_crf_tagging/bert_crf_tagging_seed_12",
                        help="Directory of the model files.")
    parser.add_argument("--model_type", default="normal", choices=["normal", "crf", "date"],
                        help="Specify the type of model that is provided.")

    args = parser.parse_args()
    return args


def get_text_and_annotations_and_date(in_fp) -> (str, List[str], str):
    soup = BeautifulSoup(open(in_fp), "lxml")
    date = soup.findAll("dct")[0].findAll("timex3")[0].attrs["value"]
    content = soup.findAll("text")[0]
    annotations = []
    end = 0  # Only look in the "remaining string", by saving the previous end position
    for timex in content.findAll("timex3"):
        begin = content.text[end:].index(timex.text) + end
        end = begin + len(timex.text)
        timex_type = timex.attrs["type"]
        try:
            timex_value = timex.attrs["value"]
        except KeyError:
            timex_value = "null"
        annotations.append((begin, end, timex_type, timex_value))

    # Problems with newline chars force us to use this differently.
    text = content.text.replace("\n", " ")
    return text, annotations, date


def merge_tokens(bpe_text, bpe_predictions, id2label, tokenizer) -> List[Tuple[str, str]]:
    """
    BPEs are merged into single tokens in this step, where corresponding predictions get aggregated
    into a single token by virtue of majority voting.
    Even breaks (e.g., something like "me ##ssa ##ge | B-DATE, O, I-DURATION") will be decided by the first tag result,
    in this case "DATE" because of the tag of "me". If there is no B-tag in the current instance at all,
    the first token still decides. Note that there are no ambiguities about the B/I distinction here, since we only
    look at multi-BPE tokens, and not at tags spanning multiple *full-word* tokens.
    TODO: Note that this function gets rid of the B/I distinction for downstream tasks as well currently!
      This can be changed by not abstracting the vote to the type only, and still carrying the B-/I- prefix with it.
    :param bpe_text:
    :param bpe_predictions:
    :param id2label: Turning predicted ids back to the actual labels
    :param tokenizer: Tokenizer required to translate token ids back to the words themselves.
    :return: List of tuples containing (token, type_label) pairs.
    """
    merged_tokens = []
    prev_multi_instance = False
    current_multi_vote = []
    current_multi_token = ""
    # Iterate in reverse to immediately see when we deal with a multi-BPE instance and start voting
    for token_id, pred_id, in zip(reversed(bpe_text), reversed(bpe_predictions)):
        token = tokenizer.ids_to_tokens[int(token_id)]

        pred = id2label[int(pred_id)]

        # Skip special tokens
        if token in ("[PAD]", "[CLS]", "[SEP]"):
            continue

        # Instance for multi-BPE token
        if token.startswith("##"):
            current_multi_token = f"{token[2:]}{current_multi_token}"
            current_multi_vote.append(pred)
        else:
            # Need to merge votes
            if current_multi_token:
                current_multi_token = f"{token}{current_multi_token}"
                current_multi_vote.append(pred)
                merged_tokens.append((current_multi_token, get_vote_type(current_multi_vote)))
                current_multi_token = ""
                current_multi_vote = []
            # Previous token was single word anyways
            else:
                merged_tokens.append((token, get_pred_type(pred)))

    # Bring back into right order for later processing
    merged_tokens.reverse()
    return merged_tokens


def insert_tags_in_raw_text(raw_text: str, merged_tokens: List[Tuple[str, str]], annotation_id: int = 1):
    """
    This takes the original raw text, and iterates over it to insert the predicted tags at the right positions.
    :param raw_text:
    :param merged_tokens:
    :param annotation_id
    :return:
    """
    tagged_text = ""
    prev_tag = "O"
    current_annotation_group = ""

    for token, tag in merged_tokens:
        # If we still have the same tag, then we either just extend the annotation (not "O"), or just leave it ("O").
        if tag == prev_tag:
            if tag != "O" and tag != "[PAD]":
                current_annotation_group += f" {token}"
            continue

        else:
            # This means we're just opening an annotation, e.g., "O DATE"
            if prev_tag != "O" and prev_tag != "[PAD]":
                raw_text, tagged_text, annotation_id = place_timex_tag(raw_text, tagged_text, current_annotation_group,
                                                                       annotation_id, prev_tag)
                # Immediately store the next token, as it is also tagged, but in a different group
                if tag != "O" and prev_tag != "[PAD]":
                    current_annotation_group = token
                else:
                    current_annotation_group = ""
            else:
                current_annotation_group = token

            prev_tag = tag

    tagged_text += raw_text
    return tagged_text, annotation_id


def place_timex_tag(raw_text, tagged_text, annotation_group, annotation_id, annotation_type):
    annotation_group = preprocess_group(annotation_group)

    # Assert correct location irrespective of casing
    start_idx = raw_text.lower().find(annotation_group.lower())
    if start_idx == -1:
        import pdb
        pdb.set_trace()
        print(f"Remaining raw text: {raw_text}")
        raise ValueError(f"Could not find current annotation group \"{annotation_group}\" in text.")
    # Cannot directly write out annotation_group due to potentially different casing
    tagged_text = f"{tagged_text}{raw_text[:start_idx]}" \
                  f"<TIMEX3 tid=\"t{annotation_id}\" type=\"{annotation_type}\" value=\"\">" \
                  f"{raw_text[start_idx:start_idx+len(annotation_group)]}" \
                  f"</TIMEX3>"
    raw_text = raw_text[start_idx + len(annotation_group):]

    return raw_text, tagged_text, annotation_id+1


def preprocess_group(annotation_group: str) -> str:
    # Fix border cases of inconsistent tokens
    if "-" in annotation_group:
        annotation_group = annotation_group.replace(" - ", "-")
        annotation_group = annotation_group.replace("- ", "-")
        annotation_group = annotation_group.replace(" -", "-")
        annotation_group = annotation_group.replace("13 february-", "13 february -")
        annotation_group = annotation_group.replace("run-morning", "run - morning")
        annotation_group = annotation_group.replace("-morning", "- Morning")
        annotation_group = annotation_group.replace("next year-", "next year -")
        annotation_group = annotation_group.replace("the night of 13 february -14 february", "the night of 13 february - 14 february")
        annotation_group = annotation_group.replace("a 50-mile ( 80 km", "a 50-mile (80 km")

    if ":" in annotation_group:
        annotation_group = annotation_group.replace(" : ", ":")
        annotation_group = annotation_group.replace(": ", ":")
        annotation_group = annotation_group.replace(" :", ":")
    if "." in annotation_group:
        annotation_group = annotation_group.replace(" . ", ". ")
        annotation_group = annotation_group.replace("7. 30", "7.30")  # manual fix of
        annotation_group = annotation_group.replace(" .", ".")
        annotation_group = annotation_group.replace(". s.", ".s.")
        annotation_group = annotation_group.replace(". 2", ".2")
        annotation_group = annotation_group.replace(". 1bn", ".1bn")
        annotation_group = annotation_group.replace(". 8", ".8")
        annotation_group = annotation_group.replace(". 5", ".5")
        annotation_group = annotation_group.replace(". 6", ".6")
        annotation_group = annotation_group.replace(". d. p", ".d.p")
        annotation_group = annotation_group.replace(". 4", ".4")
        annotation_group = annotation_group.replace(". b. m.", ".b.m.")
        annotation_group = annotation_group.replace("4. 30am", "4.30am")
        annotation_group = annotation_group.replace("23. 03.2015", "23.03.2015")
        annotation_group = annotation_group.replace("feb.28", "feb. 28")
        annotation_group = annotation_group.replace(". 30am", ".30am")

    if "," in annotation_group:
        annotation_group = annotation_group.replace(" , ", ", ")
        annotation_group = annotation_group.replace(" ,", ",")
        annotation_group = annotation_group.replace("1, 460", "1,460")
        annotation_group = annotation_group.replace(", 460", ",460")
        annotation_group = annotation_group.replace(", 000", ",000")
        annotation_group = annotation_group.replace(", 445", ",445")
        annotation_group = annotation_group.replace(", 109", ",109")

    if "'" in annotation_group:
        annotation_group = annotation_group.replace("' ", "'")
        annotation_group = annotation_group.replace("day 's", "day's")
        annotation_group = annotation_group.replace("japan '", "japan'")
        annotation_group = annotation_group.replace("day '", "day'")
        annotation_group = annotation_group.replace("mussolini 's", "mussolini's")
        annotation_group = annotation_group.replace("mussolini '", "mussolini'")
        annotation_group = annotation_group.replace("five years '", "five years'")

    if "\"" in annotation_group:
        annotation_group = annotation_group.replace("\" black period", "\"black period")
        annotation_group = annotation_group.replace("black period \"", "black period\"")
        annotation_group = annotation_group.replace("period \" of the war", "period\" of the war")
        annotation_group = annotation_group.replace("the \" black", "the \"black")

    if "/" in annotation_group:
        annotation_group = annotation_group.replace("/ 2", "/2")
    return annotation_group


def get_vote_type(votes: List[str]) -> str:
    # Since Python 3.7, Counter maintains insertion order.
    # Since we want to preserve the first label in case of ties, we need to reverse the votes,
    # as we previously recorded them backwards.
    votes = [get_pred_type(vote) for vote in reversed(votes)]
    majority = Counter(votes).most_common(1)
    majority_label = majority[0][0]

    return majority_label


def get_pred_type(prediction: str) -> str:

    if prediction == "O" or prediction == "[PAD]" or prediction == "[SEP]":
        return prediction
    else:
        return prediction.split("-")[1]


def get_model_and_tokenizers(args):
    if args.model_type == "date":
        model = BERTWithDateLayerTokenClassification.from_pretrained(args.model_dir)
        date_tokenizer = DateTokenizer("./data/vocab_date.txt")
    elif args.model_type == "normal":
        model = BertForTokenClassification.from_pretrained(args.model_dir)
        date_tokenizer = None
    elif args.model_type == "crf":
        model = BertWithCRF.from_pretrained(args.model_dir)
        date_tokenizer = None
    else:
        raise ValueError("Incorrect model type specified")

    text_tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)

    return model, date_tokenizer, text_tokenizer


if __name__ == '__main__':
    args = get_args()
    model, date_tokenizer, text_tokenizer = get_model_and_tokenizers(args)
    # Guarantee existence of of output dir
    os.makedirs(args.output_dir, exist_ok=True)

    for fn in tqdm(sorted(os.listdir(args.input_dir))):
        # Ignore other file extensions
        if not fn.lower().endswith(args.file_ext):
            continue

        in_fp = os.path.join(args.input_dir, fn)
        out_fp = os.path.join(args.output_dir, fn)
        # Extract text and creation date
        raw_text, _, creation_date = get_text_and_annotations_and_date(in_fp)
        text = raw_text.split("  ")  # Since we replaced "\n" with " ", this works to detect the empty lines in between.

        if args.model_type == "date":
            creation_date = creation_date.replace("-", " ")
            # Model expects batched inputs. Date can be only the input_ids, since we don't need masks
            processed_date = torch.LongTensor(date_tokenizer([creation_date for _ in range(len(text))],
                                                             add_special_tokens=False)["input_ids"])

        # Regular text however is differently shaped, so we have to extract attention mask, too.
        processed_text = text_tokenizer(text, padding="max_length")
        input_ids = torch.LongTensor(processed_text["input_ids"])
        attention_mask = torch.LongTensor(processed_text["attention_mask"])

        with torch.no_grad():
            if args.model_type == "date":
                result = model(input_ids=input_ids, input_date_ids=processed_date, attention_mask=attention_mask)
            elif args.model_type == "normal":
                result = model(input_ids=input_ids, attention_mask=attention_mask)
            else:  # CRF
                result = model(input_ids=input_ids, attention_mask=attention_mask, inference_mode=True)
        if  args.model_type != "crf":
            classification = torch.argmax(result[0], dim=2)
        else:
            classification = result[0]

        id2label = {v: k for k, v in model.config.label2id.items()}
        final_output = ""
        annotation_id = 1
        for raw_sentence, text_slice, classification_slice in zip(text, input_ids, classification):
            if not raw_sentence:  # skip empty lines.
                continue
            merged_tokens = merge_tokens(text_slice, classification_slice, id2label, text_tokenizer)
            annotated_text, annotation_id = insert_tags_in_raw_text(raw_sentence, merged_tokens, annotation_id)
            final_output += f"{annotated_text.strip(' ')}\n\n"

        with open(in_fp) as f:
            xml = f.read()

        # Remove the original text, and replace it with our tagged version
        xml = regex.sub("<TEXT>.*</TEXT>", f"<TEXT>\n\n{final_output}</TEXT>", xml, flags=regex.DOTALL)
        # Remove reference to any events after the main text, as they cause confusion during evaluation.
        xml = regex.sub("</TEXT>.*</TimeML>", "</TEXT>\n\n</TimeML>", xml, flags=regex.DOTALL)

        with open(out_fp, "w") as f:
            f.write(xml)
