from enum import Enum
import nltk
import re
from dataclasses import dataclass

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text

nltk.download('punkt_tab')

class MaskType(Enum):
    EMAIL = "email"
    PHONE = "phone"
    IP = "ip"

@dataclass
class Models:
    language: str = "local-shared-data/lid.176.bin"
    nsfw: str = "local-shared-data/jigsaw_fasttext_bigrams_nsfw_final.bin"
    hate: str = "local-shared-data/jigsaw_fasttext_bigrams_hatespeech_final.bin"


def extract_text_from_html(html_bytes: bytes) -> str | None:
    encoding = detect_encoding(html_bytes)
    decoded_html = html_bytes.decode(encoding)
    return extract_plain_text(decoded_html)

def get_prediction(text: str, model_path: str) -> tuple[str, float]:
    text = ' '.join([el.strip() for el in text.split('\n')])
    model = fasttext.load_model(model_path)
    pred = model.predict(text)
    return pred[0][0].split('__')[-1], pred[1][0]

def mask_specific_text(text: str, type: MaskType) -> tuple[str, int]:
    if type == MaskType.EMAIL:
        regex = r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
        repl = "|||EMAIL_ADDRESS|||"
    elif type == MaskType.PHONE:
        regex = r"(?<!\w)(?:\+?\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)|\d{2,4})[\s.-]?\d{3,5}[\s.-]?\d{3,5}(?!\w)"
        repl = "|||PHONE_NUMBER|||"
    elif type == MaskType.IP:
        regex = re.compile(
            r"\b(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)"
            r"(?:\.(?:25[0-5]|2[0-4]\d|1\d\d|[1-9]?\d)){3}\b"
        )
        repl = "|||IP_ADDRESS|||"

    return re.subn(regex, repl, text)

def run_quality_filter_gopher(text: str) -> bool:
    words = nltk.word_tokenize(text)
    num_words = len(words)
    mean_word_len = sum([len(w) for w in words])/num_words
    one_alpha_words = sum([1 for w in words if bool(re.search(r"[A-Za-z]", w))])
    num_lines = len(text.splitlines())
    ellipsis_lines = sum([1 for line in text.splitlines() if line[-3:] == "..."])
    if num_words < 50 or num_words > 100000:
        return False
    if mean_word_len < 3 or mean_word_len > 10:
        return False
    if one_alpha_words/num_words < 0.8:
        return False
    if ellipsis_lines/num_lines > 0.3:
        return False
    return True

if __name__ == "__main__":
    with open('local-shared-data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz', 'rb') as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            data_bytes = record.reader.read()
            data_str = extract_text_from_html(data_bytes)
            lang = get_prediction(data_str, Models.language)
            if lang[0] == "en":
                nsfw = get_prediction(data_str, Models.nsfw)
                hate = get_prediction(data_str, Models.hate)
                print(nsfw, hate)
                print(data_str)

