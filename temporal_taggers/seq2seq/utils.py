
import numpy as np
from bs4 import BeautifulSoup


class Metrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.result = {}

    def compute_metrics(self, eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        if preds.shape == labels.shape:
            preds = np.where(labels != -100, preds, self.tokenizer.pad_token_id)
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Metrics that will be computed
        date_tp = 0
        type_tp = 0
        both_tp = 0
        date_retrieved = 0
        type_retrieved = 0
        both_retrieved = 0
        total_timex = 0

        exact_match_tp = 0
        partial_match_tp = 0
        exact_match_fn = 0
        partial_match_fn = 0
        exact_match_fp = 0
        partial_match_fp = 0

        decoded_preds, decoded_labels = self.postprocess_text(decoded_preds, decoded_labels)

        # TODO: [DA: 2021-11-21] Comment and clean up loop.
        for pred, label in zip(decoded_preds, decoded_labels):
            if "timex3" in label[0]:
                pred = pred[:len(label[0]) + 1]
                pred = pred.replace(" < / timex3  ", " </timex3").replace("< timex3 ", "<timex3 ").replace(
                    "< / timex3 >",
                    "</timex3>").replace(
                    "/ timex3 >", "</timex3>")
                label = label[0].replace(" < / timex3  ", " </timex3").replace("< timex3 ", "<timex3 ").replace(
                    "< / timex3 >", "</timex3>")
                pred_soup = BeautifulSoup(pred, "lxml")
                label_soup = BeautifulSoup(label, "lxml")
                if "timex3" in pred and "<timex3" not in pred:
                    pred.replace("timex3", "<timex3")
                if "/timex3>" in pred and not "</timex3>":
                    pred.replace("/timex3>", "</timex3>")

                timex_labels = label_soup.findAll("timex3")
                if "<timex3" in pred:
                    timex_preds = pred_soup.findAll("timex3")

                    for i in range(len(timex_labels)):
                        for j in range(i, len(timex_preds)):
                            flag = False
                            cleaned_text = timex_preds[j].text.replace("<", "").replace(">", "").replace("\"", "").strip()
                            if len(cleaned_text.split(" ")) > 1 and cleaned_text.split(" ")[0] == cleaned_text.split(" ")[1]:
                                cleaned_text = ' '.join(cleaned_text.split(" ")[1:])

                            if cleaned_text == timex_labels[i].text.strip():
                                exact_match_tp = exact_match_tp + 1
                                flag = True

                            if any([text in timex_labels[i].text.strip() for text in cleaned_text.split(" ")]):
                                partial_match_tp = partial_match_tp + 1
                                flag = True

                            if not flag:
                                continue

                            if "value" in timex_preds[j].attrs:
                                date_retrieved = date_retrieved + 1
                                if timex_preds[j].attrs["value"].strip() == timex_labels[i].attrs["value"].strip():
                                    date_tp = date_tp + 1  # retrieved docs intersection with relevant docs

                            if "type" in timex_preds[j].attrs:
                                type_retrieved = type_retrieved + 1
                                if timex_preds[j].attrs["type"].strip() == timex_labels[i].attrs["type"].strip():
                                    type_tp = type_tp + 1

                            if "type" in timex_preds[j].attrs and "value" in timex_preds[j].attrs:
                                both_retrieved = both_retrieved + 1
                                if timex_preds[j].attrs["value"].strip() == timex_labels[i].attrs["value"].strip() and \
                                   timex_labels[i].attrs["type"].strip() == timex_preds[j].attrs["type"].strip():
                                    both_tp = both_tp + 1
                            if flag:
                                break

                    if partial_match_tp < len(timex_labels):
                        partial_match_fn = partial_match_fn + len(timex_labels) - partial_match_tp

                    if exact_match_tp < len(timex_labels):
                        exact_match_fn = exact_match_fn + len(timex_labels) - exact_match_tp

                    if len(timex_preds) > len(timex_labels):
                        exact_match_fp = exact_match_fp + len(timex_preds) - exact_match_tp
                        partial_match_fp = partial_match_fp + len(timex_preds) - exact_match_tp
                else:
                    partial_match_fn = partial_match_fn + len(timex_labels)
                    exact_match_fn = exact_match_fn + len(timex_labels)

                total_timex = float(total_timex + len(timex_labels))  # relevant docs

        # Accuracy scores
        self.result["date_acurracy"] = date_tp / total_timex if total_timex != 0 else 0
        self.result["type_accuracy"] = type_tp / total_timex if total_timex != 0 else 0
        self.result["both_accuracy"] = both_tp / total_timex if total_timex != 0 else 0
        self.result["exact_accuracy"] = exact_match_tp / total_timex if total_timex != 0 else 0
        self.result["partial_accuracy"] = partial_match_tp / total_timex if total_timex != 0 else 0

        # Precision scores
        self.result["exact_precision"] = (exact_match_tp / (exact_match_tp + exact_match_fp)) if (exact_match_tp + exact_match_fp) != 0 else 0
        self.result["partial_precision"] = (partial_match_tp / (partial_match_tp + partial_match_fp)) if (partial_match_tp + partial_match_fp) != 0 else 0
        self.result["date_precision"] = (date_tp / date_retrieved) if date_retrieved != 0 else 0
        self.result["type_precision"] = (type_tp / type_retrieved) if type_retrieved != 0 else 0
        self.result["both_precision"] = (both_tp / both_retrieved) if both_retrieved != 0 else 0

        # Recall scores
        self.result["exact_recall"] = (exact_match_tp / (exact_match_tp + exact_match_fn)) if (exact_match_tp + exact_match_fn) != 0 else 0
        self.result["partial_recall"] = (partial_match_tp / (partial_match_tp + partial_match_fn)) if (partial_match_tp + partial_match_fn) != 0 else 0
        # FIXME: [DA: 2021-11-21] This seems to be the same as accuracy. Is this intentional?
        self.result["date_recall"] = (date_tp / total_timex) if total_timex != 0 else 0
        self.result["type_recall"] = (type_tp / total_timex) if total_timex != 0 else 0
        self.result["both_recall"] = (both_tp / total_timex) if total_timex != 0 else 0

        # F1 scores
        self.result["exact_f1"] = ((2 * (self.result["exact_precision"] * self.result["exact_recall"])) / (self.result["exact_precision"] + self.result["exact_recall"])) if (self.result["exact_precision"] + self.result["exact_recall"]) != 0 else 0
        self.result["partial_f1"] = ((2 * (self.result["partial_precision"] * self.result["partial_recall"])) / (self.result["partial_precision"] + self.result["partial_recall"])) if (self.result["partial_precision"] + self.result["partial_recall"]) != 0 else 0
        self.result["date_f1"] = ((2 * (self.result["date_precision"] * self.result["date_recall"])) / (self.result["date_precision"] + self.result["date_recall"])) if (self.result["date_precision"] + self.result["date_recall"]) != 0 else 0
        self.result["type_f1"] = ((2 * (self.result["type_precision"] * self.result["type_recall"])) / (self.result["type_precision"] + self.result["type_recall"])) if (self.result["type_precision"] + self.result["type_recall"]) != 0 else 0
        self.result["both_f1"] = ((2 * (self.result["both_precision"] * self.result["both_recall"])) / (self.result["both_precision"] + self.result["both_recall"])) if (self.result["both_precision"] + self.result["both_recall"]) != 0 else 0

        return self.result

    @staticmethod
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels


class DataProcessor:
    def __init__(self, tokenizer, text_column, target_column, prefix, max_source_length, max_target_length, padding,
                 date_column=None):
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
