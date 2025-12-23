# from google.cloud import vision

    # client = vision.ImageAnnotatorClient()


    # def ocr_image_bytes(image_bytes: bytes) -> dict:
    #     image = vision.Image(content=image_bytes)
    #     response = client.text_detection(image=image)

    #     if response.error.message:
    #         raise RuntimeError(response.error.message)

    #     text_annotation = response.full_text_annotation
    #     text = text_annotation.text or ""
    #     confidence = (
    #         sum(page.confidence for page in text_annotation.pages)
    #         / max(len(text_annotation.pages), 1)
    #     )

    #     return {"text": text, "confidence": confidence}
    # app/services/vision_api.py
import pytesseract
from PIL import Image
import io

def ocr_image_bytes(image_bytes: bytes) -> dict:
    image = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(image)
    return {
        "text": text.strip(),
        "confidence": None,
    }

