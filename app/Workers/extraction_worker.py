# app/workers/extraction_worker.py

from typing import Dict
from google.cloud import vision

from app.services.vision_api import ocr_image_bytes
from app.utils.text_cleaner import clean_text, format_ocr_result

_vision_client = vision.ImageAnnotatorClient()


def extract_from_image(image_bytes: bytes) -> Dict:
   
    result = ocr_image_bytes(image_bytes)

    text = clean_text(result.get("text", ""))
    confidence = result.get("confidence", 0.0)

    return format_ocr_result(text, confidence)


def extract_from_pdf(pdf_bytes: bytes) -> Dict:
    
    input_config = vision.InputConfig(
        content=pdf_bytes,
        mime_type="application/pdf",
    )

    request = vision.AnnotateFileRequest(
        input_config=input_config,
        features=[vision.Feature(type_=vision.Feature.Type.DOCUMENT_TEXT_DETECTION)],
    )

    response = _vision_client.batch_annotate_files(requests=[request])

    if response.responses[0].error.message:
        raise RuntimeError(response.responses[0].error.message)

    pages = response.responses[0].responses
    full_text = " ".join(
        page.full_text_annotation.text
        for page in pages
        if page.full_text_annotation.text
    )

    text = clean_text(full_text)
    confidence = 0.0

    return format_ocr_result(text, confidence)
