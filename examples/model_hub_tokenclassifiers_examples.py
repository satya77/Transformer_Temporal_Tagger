from transformers import AutoTokenizer, BertForTokenClassification
from temporal_models.BERTWithDateLayerTokenClassification import BERTWithDateLayerTokenClassification
from temporal_models.NumBertTokenizer import NumBertTokenizer
from temporal_models.BERTWithCRF import BERT_CRF_NER
from evaluation.classifier_generate_tempeval_data import merge_tokens, insert_tags_in_raw_text
import torch

if __name__ == "__main__":
    model_type = "normal"  # change the model type here to try different models

    input_texts = ["I lived in New York for 10 years."]
    input_texts += ["Cumbre Vieja last erupted in 1971 and in 1949."]
    input_texts += ["The club's founding date, 15 January, was intentional."]
    input_texts += ["Officers were called to the house at 07:25 BST on Sunday after concerns were raised about the people living there."]
    input_texts += ["Police were first called to the scene just after 7.25am this morning, Sunday, September 19, and have confirmed they will continue to remain in the area for some time."]

    if model_type == "date":
        model = BERTWithDateLayerTokenClassification.from_pretrained(
            "satyaalmasian/temporal_tagger_DATEBERT_tokenclassifier")
        date_tokenizer = NumBertTokenizer("../data/vocab_date.txt")
        processed_date = torch.LongTensor(date_tokenizer(["2020 2 28" for _ in range(len(input_texts))],
                                                         add_special_tokens=False)["input_ids"])  # some random date
    elif model_type == "normal":
        model = BertForTokenClassification.from_pretrained("satyaalmasian/temporal_tagger_BERT_tokenclassifier")
        date_tokenizer = None
    elif model_type == "crf":
        model = BERT_CRF_NER.from_pretrained("satyaalmasian/temporal_tagger_BERTCRF_tokenclassifier")
        date_tokenizer = None
    else:
        raise ValueError("Incorrect model type specified")

    text_tokenizer = AutoTokenizer.from_pretrained("satyaalmasian/temporal_tagger_BERT_tokenclassifier", use_fast=False)

    id2label = {v: k for k, v in model.config.label2id.items()}
    annotation_id = 1
    for input_text in input_texts:
        processed_text = text_tokenizer(input_text, return_tensors="pt")

        if model_type == "date":
            processed_text["input_date_ids"] = processed_date
            result = model(**processed_text)
        elif model_type == "normal":
            result = model(**processed_text)
        else:
            processed_text["inference_mode"] = True
            result = model(**processed_text)

        if model_type != "crf":
            classification = torch.argmax(result[0], dim=2)
        else:
            classification = result[0]
        merged_tokens = merge_tokens(processed_text["input_ids"][0], classification[0], id2label, text_tokenizer)
        annotated_text, annotation_id = insert_tags_in_raw_text(input_text, merged_tokens, annotation_id)
        print(annotated_text)
