
from fastwarc.warc import ArchiveIterator, WarcRecordType
from resiliparse.parse.encoding import detect_encoding
from resiliparse.extract.html2text import extract_plain_text


def extract_text_from_html(html_bytes: bytes) -> str | None:
    encoding = detect_encoding(html_bytes)
    decoded_html = html_bytes.decode(encoding)
    return extract_plain_text(decoded_html)


if __name__ == "__main__":
    with open('local-shared-data/CC-MAIN-20250417135010-20250417165010-00065.warc.gz', 'rb') as f:
        for record in ArchiveIterator(f, record_types=WarcRecordType.response):
            data_bytes = record.reader.read()
            data_str = extract_text_from_html(data_bytes)
            print(data_str)
            break