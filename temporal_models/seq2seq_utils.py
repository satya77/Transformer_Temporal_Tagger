import numpy as np
from bs4 import BeautifulSoup
import re

class metrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        if preds.shape == labels.shape:
            preds = np.where(labels != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Some simple post-processing
        date_tp = 0
        type_tp = 0
        both_tp = 0
        date_retrieved = 0
        type_retrieved = 0
        both_retrieved = 0
        result = {}
        total_timex = 0

        exact_match_tp = 0
        partical_match_tp = 0
        exact_match_fn = 0
        partical_match_fn = 0
        exact_match_fp = 0
        partical_match_fp = 0

        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)
        for pred, lable in zip(decoded_preds, decoded_labels):
            if "timex3" in lable[0]:
                pred = pred[:len(lable[0]) + 1]
                pred = pred.replace(" < / timex3  ", " </timex3").replace("< timex3 ", "<timex3 ").replace(
                    "< / timex3 >",
                    "</timex3>").replace(
                    "/ timex3 >", "</timex3>")
                lable = lable[0].replace(" < / timex3  ", " </timex3").replace("< timex3 ", "<timex3 ").replace(
                    "< / timex3 >", "</timex3>")
                pred_soup = BeautifulSoup(pred, "lxml")
                label_soup = BeautifulSoup(lable, "lxml")
                if "timex3" in pred and "<timex3" not in pred:
                    pred.replace("timex3", "<timex3")
                if "/timex3>" in pred and not "</timex3>":
                    pred.replace("/timex3>", "</timex3>")

                timex_labels = label_soup.findAll("timex3")
                if "<timex3" in pred:
                    try:
                        timex_preds = pred_soup.findAll("timex3")

                        for i in range(len(timex_labels)):
                            for j in range(i, len(timex_preds)):
                                flag = False
                                cleand_text = timex_preds[j].text.replace("<", "").replace(">", "").replace("\"", "").strip()
                                if len (cleand_text.split(" "))>1 and  cleand_text.split(" ")[0]==cleand_text.split(" ")[1]:
                                    cleand_text = ' '.join(cleand_text.split(" ")[1:])

                                if cleand_text == timex_labels[i].text.strip():
                                    exact_match_tp = exact_match_tp + 1
                                    flag = True

                                if any([text in timex_labels[i].text.strip() for text in cleand_text.split(" ")]):
                                    partical_match_tp = partical_match_tp + 1
                                    flag = True

                                if not flag:
                                    continue

                                if "value" in timex_preds[j].attrs:
                                    # retrieved
                                    date_retrieved = date_retrieved + 1
                                    if (timex_preds[j].attrs["value"].strip() == timex_labels[i].attrs["value"].strip()):
                                        date_tp = date_tp + 1  # retrieved docs intersection with relevant docs

                                if "type" in timex_preds[j].attrs:
                                    type_retrieved = type_retrieved + 1
                                    if timex_preds[j].attrs["type"].strip() == timex_labels[i].attrs["type"].strip():
                                        type_tp = type_tp + 1
                                if ("type" in timex_preds[j].attrs and "value" in timex_preds[j].attrs):
                                    both_retrieved = both_retrieved + 1
                                    if timex_preds[j].attrs["value"].strip() == timex_labels[i].attrs[
                                        "value"].strip() and \
                                            timex_labels[i].attrs["type"].strip() == timex_preds[j].attrs["type"].strip():
                                        both_tp = both_tp + 1
                                if flag:
                                    break

                        if partical_match_tp < len(timex_labels):
                            partical_match_fn = partical_match_fn + len(timex_labels) - partical_match_tp

                        if exact_match_tp < len(timex_labels):
                            exact_match_fn = exact_match_fn + len(timex_labels) - exact_match_tp

                        if len(timex_preds) > len(timex_labels):
                            exact_match_fp = exact_match_fp + len(timex_preds) - exact_match_tp
                            partical_match_fp = partical_match_fp + len(timex_preds) - exact_match_tp
                    except:
                        import pdb
                        pdb.set_trace()
                else:
                    partical_match_fn = partical_match_fn + len(timex_labels)
                    exact_match_fn = exact_match_fn + len(timex_labels)

                total_timex = total_timex + len(timex_labels)  # relevant docs

        result["date_acurracy"] = (date_tp / float(total_timex)) if total_timex != 0 else 0
        result["type_accuracy"] = (type_tp / float(total_timex)) if total_timex != 0 else 0
        result["both_accuracy"] = (both_tp / float(total_timex)) if total_timex != 0 else 0
        result["exact_accuracy"] = (exact_match_tp / float(total_timex)) if total_timex != 0 else 0
        result["partial_accuracy"] = (partical_match_tp / float(total_timex)) if total_timex != 0 else 0
        result["exact_precision"] = (exact_match_tp / float(exact_match_tp + exact_match_fp)) if (
                                                                                                             exact_match_tp + exact_match_fp) != 0 else 0
        result["partial_precision"] = (partical_match_tp / float(partical_match_tp + partical_match_fp)) if (
                                                                                                                        partical_match_tp + partical_match_fp) != 0 else 0
        result["date_precision"] = (date_tp / float(date_retrieved)) if (date_retrieved) != 0 else 0
        result["type_precision"] = (type_tp / float(type_retrieved)) if (type_retrieved) != 0 else 0
        result["both_precision"] = (both_tp / float(both_retrieved)) if (both_retrieved) != 0 else 0
        result["exact_recall"] = (exact_match_tp / float(exact_match_tp + exact_match_fn)) if (
                                                                                                          exact_match_tp + exact_match_fn) != 0 else 0
        result["partial_recall"] = (partical_match_tp / float(partical_match_tp + partical_match_fn)) if (
                                                                                                                     partical_match_tp + partical_match_fn) != 0 else 0
        result["date_recall"] = (date_tp / float(total_timex)) if total_timex != 0 else 0
        result["type_recall"] = (type_tp / float(total_timex)) if total_timex != 0 else 0
        result["both_recall"] = (both_tp / float(total_timex)) if total_timex != 0 else 0
        result["exact_f1"] = ((2 * (result["exact_precision"] * result["exact_recall"])) / float(
            result["exact_precision"] + result["exact_recall"])) if (result["exact_precision"] + result[
            "exact_recall"]) != 0 else 0
        result["partial_f1"] = ((2 * (result["partial_precision"] * result["partial_recall"])) / float(
            result["partial_precision"] + result["partial_recall"])) if (result["partial_precision"] + result[
            "partial_recall"]) != 0 else 0
        result["date_f1"] = ((2 * (result["date_precision"] * result["date_recall"])) / float(
            result["date_precision"] + result["date_recall"])) if (result["date_precision"] + result[
            "date_recall"]) != 0 else 0
        result["type_f1"] = ((2 * (result["type_precision"] * result["type_recall"])) / float(
            result["type_precision"] + result["type_recall"])) if (result["type_precision"] + result[
            "type_recall"]) != 0 else 0
        result["both_f1"] = ((2 * (result["both_precision"] * result["both_recall"])) / float(
            result["both_precision"] + result["both_recall"])) if (result["both_precision"] + result[
            "both_recall"]) != 0 else 0

        return result

   

class DataProcessor:
    def __init__(self, tokenizer, text_column, target_column, prefix, max_source_length, max_target_length, padding,date_column=None):
        self.tokenizer = tokenizer
        self.text_column = text_column
        self.target_column = target_column
        self.prefix = prefix
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.padding = padding
        self.date_column = date_column



    def process_data_to_model_inputs_encoder_decoder(self, batch):
        # tokenize the inputs and labels

        inputs = self.tokenizer(batch[self.text_column], padding=self.padding, truncation=True,
                                max_length=self.max_source_length)
        outputs = self.tokenizer(batch[self.target_column], padding=self.padding, truncation=True,
                                 max_length=self.max_target_length)

        batch["input_ids"] = inputs.input_ids
        batch["attention_mask"] = inputs.attention_mask
        batch["decoder_input_ids"] = outputs.input_ids
        batch["decoder_attention_mask"] = outputs.attention_mask
        batch["labels"] = outputs.input_ids.copy()
        batch["labels"] = [[-100 if token == self.tokenizer.pad_token_id else token for token in labels] for labels in
                           batch["labels"]]

        return batch


