import io
from typing import Dict

from PIL import Image
import pytesseract
import pytesseract as tess
from pytesseract import Output
from pypdf import PdfReader

from app.utils.text_cleaner import clean_text


# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def extract_from_image(image_bytes: bytes) -> Dict:
    """
    OCR image and return cleaned text + average confidence
    """
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    data = tess.image_to_data(image, output_type=Output.DICT)

    words = []
    confidences = []

    for text, conf in zip(data["text"], data["conf"]):
        if text.strip() and conf != "-1":
            words.append(text)
            confidences.append(int(conf))

    raw_text = " ".join(words)
    cleaned_text = clean_text(raw_text)

    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "text": cleaned_text,
        "confidence": round(avg_conf / 100, 2),
    }


def extract_from_pdf(pdf_bytes: bytes) -> Dict:
    """
    Extract text from PDF (digital PDFs only).
    Confidence is None because PDFs don't provide OCR confidence.
    """
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text = []

    for page in reader.pages:
        text = page.extract_text()
        if text:
            pages_text.append(text)

    raw_text = "\n".join(pages_text)
    cleaned_text = clean_text(raw_text)

    return {
        "text": cleaned_text,
        "confidence": None,
    }
