import base64
import os

import requests


GEMINI_VISION_MODELS = [
    os.getenv("GEMINI_VISION_MODEL", "").strip(),
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-2.5-flash-lite",
]
GEMINI_VISION_MODELS = [model for model in GEMINI_VISION_MODELS if model]


def detect_mime(content_type: str = "") -> str:
    content_type = (content_type or "").lower()
    if "png" in content_type:
        return "image/png"
    if "webp" in content_type:
        return "image/webp"
    if "gif" in content_type:
        return "image/gif"
    return "image/jpeg"


def read_image_with_gemini(image_path_or_url: str, gemini_key: str = None, is_url: bool = False) -> str:
    key = gemini_key or os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    if is_url:
        response = requests.get(image_path_or_url, timeout=20)
        response.raise_for_status()
        img_data = response.content
        mime = detect_mime(response.headers.get("Content-Type", "image/jpeg"))
    else:
        with open(image_path_or_url, "rb") as f:
            img_data = f.read()
        mime = "image/jpeg"

    img_b64 = base64.b64encode(img_data).decode("utf-8")
    payload = {
        "contents": [
            {
                "parts": [
                    {"inlineData": {"mimeType": mime, "data": img_b64}},
                    {
                        "text": (
                            "You are reading an image for a personal knowledge bot. "
                            "First identify the visual form: screenshot, website screenshot, social post, "
                            "scholarship notice, academic notice, AI tool page, course poster, photo, or other. "
                            "Then extract all visible Bengali and English text exactly. "
                            "List visible URLs, website names, organization names, tool names, dates, deadlines, "
                            "and any suggested action. Do not guess beyond the image."
                        )
                    },
                ]
            }
        ]
    }

    errors = []
    for model in GEMINI_VISION_MODELS:
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
        r = requests.post(api_url, json=payload, timeout=40)
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        errors.append(f"{model}: {r.status_code} {r.text[:120]}")

    raise RuntimeError("Gemini Vision failed for all models: " + " | ".join(errors))
