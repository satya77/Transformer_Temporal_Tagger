from bs4 import BeautifulSoup
from transformers import AutoTokenizer, EncoderDecoderModel

from temporal_taggers.evaluation import clean_predictions


def find_timex_in_text(timex_preds, input_text, model_type):
    if model_type == "bert":
        original_paragraph = input_text.lower()
    else:
        original_paragraph = input_text
    end_previous_timex = 0
    previous_timex_cleaned_text = ""
    new_text = ""
    index = 0
    for timex in timex_preds:
        cleaned_text = timex.text.replace("<", "").replace(">", "").replace("\"", "").strip()
        # sometimes the cleaned text has "leftovers"
        if cleaned_text.startswith("- "):
            cleaned_text = cleaned_text[2:]

        if len(cleaned_text) < 2:
            continue

        beginning_timex = original_paragraph[end_previous_timex:].find(cleaned_text)
        if cleaned_text == "day" and beginning_timex != -1 and \
                original_paragraph[beginning_timex - 2:beginning_timex] == "to":
            cleaned_text = "today"
            beginning_timex = original_paragraph[end_previous_timex:].find(cleaned_text)

        # if the model predicted a full year instead of the last two digits
        if beginning_timex == -1 and len(cleaned_text) == 4 and cleaned_text.isdigit():
            beginning_timex = original_paragraph[end_previous_timex:].find(cleaned_text[2:])
            cleaned_text = cleaned_text[2:].strip()

        # if the model predicted full year with an extra repetition
        if beginning_timex == -1 and len(cleaned_text) == 6 and cleaned_text.isdigit():
            beginning_timex = original_paragraph[end_previous_timex:].find(cleaned_text[:-2])
            cleaned_text = cleaned_text[:-2].strip()

        # if the first word is repeating
        elif beginning_timex == -1 and len(cleaned_text.split(" ")) > 1 and \
                cleaned_text.split(" ")[0] == cleaned_text.split(" ")[1]:
            cleaned_text = ' '.join(cleaned_text.split(" ")[:-1])
            beginning_timex = original_paragraph[end_previous_timex:].find(cleaned_text)

        # if the first and last word is repeating
        elif beginning_timex == -1 and len(cleaned_text.split(" ")) > 1 and \
                cleaned_text.split(" ")[0] == cleaned_text.split(" ")[-1]:
            cleaned_text = ' '.join(cleaned_text.split(" ")[1:])
            beginning_timex = original_paragraph[end_previous_timex:].find(cleaned_text)
        # if its single word separated by "-"
        elif beginning_timex == -1 and len(cleaned_text.split(" ")) < 2 and len(cleaned_text.split("-")) > 1:
            for word in cleaned_text.split("-"):
                if word in original_paragraph[end_previous_timex:]:
                    cleaned_text = word
                    beginning_timex = original_paragraph[end_previous_timex:].find(cleaned_text)
                    break
        # more than one words the first one is a digit
        elif beginning_timex == -1 and len(cleaned_text.split(" ")) < 2 and len(cleaned_text) > 2 and \
                not cleaned_text[:1].isdigit() and cleaned_text[-1].isdigit():
            word = cleaned_text[:-1]
            if word.lower() in original_paragraph[end_previous_timex:].lower():
                cleaned_text = word
                beginning_timex = original_paragraph[end_previous_timex:].lower().find(cleaned_text.lower())
                break;
        # if its just a single word
        elif beginning_timex == -1 and len(cleaned_text.split(" ")) < 2 and len(cleaned_text) > 2 and \
                not cleaned_text[0].isdigit() and cleaned_text[-1].isdigit():
            for i in range(2, len(cleaned_text)):
                word = cleaned_text[:i]
                if " " + word + " " in original_paragraph[end_previous_timex:] or \
                        " " + word + "." in original_paragraph[end_previous_timex:] or \
                        " " + word + "," in original_paragraph[end_previous_timex:]:
                    cleaned_text = word
                    beginning_timex = original_paragraph[end_previous_timex:].find(cleaned_text)
                    break;

        # if its just a single word ending with digits
        if beginning_timex == -1 and len(cleaned_text.split(" ")) < 2:
            for i in range(2, len(cleaned_text)):
                word = cleaned_text[:i]
                if " " + word + " " in original_paragraph[end_previous_timex:] or \
                        " " + word + "." in original_paragraph[end_previous_timex:] or \
                        " " + word + "," in original_paragraph[end_previous_timex:]:
                    cleaned_text = word
                    beginning_timex = original_paragraph[end_previous_timex:].find(cleaned_text)
                    break;
        # if you can not find it, see if you can match the first word in the multi word one
        if beginning_timex == -1 and len(cleaned_text.split(" ")) > 1:
            for word in cleaned_text.split(" "):
                if word in original_paragraph[end_previous_timex:] and word not in ["a", "-", ".", "the",
                                                                                    "in", "then", "'s",
                                                                                    "have", "at", "be"]:
                    cleaned_text = word
                    beginning_timex = original_paragraph[end_previous_timex:].find(cleaned_text)
                    break

        if beginning_timex == -1 and cleaned_text.lower() in original_paragraph[
                                                             end_previous_timex:].lower():
            beginning_timex = original_paragraph[end_previous_timex:].lower().find(cleaned_text.lower())

        # avoid tag repetition
        if cleaned_text == previous_timex_cleaned_text:
            continue

        previous_timex_cleaned_text = cleaned_text

        # if there is still no match, just forget it.
        if beginning_timex == -1:
            continue

        index = index + 1
        beginning_timex = beginning_timex + end_previous_timex
        # if the word ended with one of these symbols do not put a space after timex tag
        if original_paragraph[beginning_timex - 1:beginning_timex] in ["\n", "'", "-", ",", "\"", "("] or \
                original_paragraph[beginning_timex - 1:beginning_timex].isdigit():
            new_text += f'{input_text[end_previous_timex:beginning_timex]}<TIMEX3 tid="t{index + 1}" ' \
                        f'type="{timex.attrs["type"].upper()}" ' \
                        f'value="{timex.attrs["value"].strip().replace("</timex3>", "").replace("<", "").replace(">", "").replace(" ", "").upper()}">{input_text[beginning_timex:beginning_timex + len(cleaned_text)]}' \
                        f'</TIMEX3>'

        else:  # otherwise put a space
            new_text += f'{input_text[end_previous_timex:beginning_timex]} <TIMEX3 tid="t{index + 1}" ' \
                        f'type="{timex.attrs["type"].upper()}" ' \
                        f'value="{timex.attrs["value"].strip().replace("</timex3>", "").replace("<", "").replace(">", "").replace(" ", "").upper()}">{input_text[beginning_timex:beginning_timex + len(cleaned_text)]}' \
                        f'</TIMEX3>'

        end_previous_timex = beginning_timex + len(cleaned_text)

    new_text += input_text[end_previous_timex:]
    return new_text


if __name__ == "__main__":
    model_type = "roberta"
    tokenizer = AutoTokenizer.from_pretrained("satyaalmasian/temporal_tagger_roberta2roberta")
    model = EncoderDecoderModel.from_pretrained("satyaalmasian/temporal_tagger_roberta2roberta")

    # --- if you want to use the bert model, uncomment the following lines
    # model_type="bert"
    # tokenizer = AutoTokenizer.from_pretrained("satyaalmasian/temporal_tagger_bert2bert")
    # model = EncoderDecoderModel.from_pretrained("satyaalmasian/temporal_tagger_bert2bert")

    input_texts = ["I lived in New York for 10 years."]
    input_texts += ["Cumbre Vieja last erupted in 1971 and in 1949."]
    input_texts += ["The club's founding date, 15 January, was intentional."]
    input_texts += ["Police were first called to the scene just after 7.25am this morning, Sunday, September 19, "
                    "and have confirmed they will continue to remain in the area for some time."]

    for input_text in input_texts:
        model_inputs = tokenizer(input_text, truncation=True, return_tensors="pt")
        out = model.generate(**model_inputs)
        decoded_preds = tokenizer.batch_decode(out, skip_special_tokens=True)
        pred_soup = BeautifulSoup(clean_predictions(decoded_preds[0]), "lxml")
        timex_preds = pred_soup.findAll("timex3")
        new_text = find_timex_in_text(timex_preds, input_text, model_type)
        print(new_text)
