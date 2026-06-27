from enum import Enum
import re

import fasttext
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text

class MaskType(Enum):
    EMAIL = "email"
    PHONE = "phone"
    IP = "ip"


def extract_text_from_html(html_bytes: bytes) -> str | None:
    encoding = detect_encoding(html_bytes)
    decoded_html = html_bytes.decode(encoding)
    return extract_plain_text(decoded_html)

def identify_language(text: str) -> tuple[str, float]:
    text = ' '.join([el.strip() for el in text.split('\n')])
    model = fasttext.load_model("local-shared-data/lid.176.bin")
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


if __name__ == "__main__":
    with open('local-shared-data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz', 'rb') as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            data_bytes = record.reader.read()
            data_str = extract_text_from_html(data_bytes)
            pred = identify_language(data_str)
            print(pred)
