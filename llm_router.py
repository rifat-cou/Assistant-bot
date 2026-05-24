import os
import subprocess

import requests

from classifier import build_categorize_prompt, clean_json_response, normalize_ai_data
from gemini_vision import read_image_with_gemini


class LLMRouter:
    """Shared Groq + DeepSeek + Gemini helper for both Discord and Telegram."""

    def __init__(self, groq_key=None, deepseek_key=None, gemini_key=None):
        self.groq_key = groq_key or os.getenv("GROQ_API_KEY")
        self.deepseek_key = deepseek_key or os.getenv("DEEPSEEK_API_KEY")
        self.gemini_key = gemini_key or os.getenv("GEMINI_API_KEY")

    def call_deepseek(self, prompt: str, max_tokens: int = 700, temperature: float = 0.1) -> str:
        if not self.deepseek_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {self.deepseek_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=30,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError(f"DeepSeek {r.status_code}: {r.text[:150]}")

    def call_groq(self, prompt: str, max_tokens: int = 700, temperature: float = 0.1) -> str:
        if not self.groq_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.groq_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": temperature,
            },
            timeout=25,
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError(f"Groq {r.status_code}: {r.text[:150]}")

    def categorize(self, text: str, source: str = "", url: str = "") -> dict:
        prompt = build_categorize_prompt(text, source=source, url=url)
        providers = []
        if self.deepseek_key:
            providers.append(("DeepSeek", self.call_deepseek))
        if self.groq_key:
            providers.append(("Groq", self.call_groq))

        for provider_name, provider_fn in providers:
            try:
                raw = provider_fn(prompt, max_tokens=700, temperature=0.1)
                data = normalize_ai_data(clean_json_response(raw), source=source, text=text)
                data["_llm_used"] = provider_name
                return data
            except Exception:
                continue

        data = normalize_ai_data(
            {
                "title": f"Saved content from {source}",
                "category": "Unknown",
                "content_type": "Unknown",
                "confidence": 0.0,
                "needs_review": True,
                "summary_en": text[:240],
                "summary_bn": "বিষয়বস্তু সংরক্ষিত হয়েছে।",
            },
            source=source,
            text=text,
        )
        data["_llm_used"] = "fallback"
        return data

    def answer(self, message: str, playful: bool = False) -> str:
        tone = "friendly, playful, and casual" if playful else "helpful, concise, and practical"
        prompt = f"""You are Marof's personal assistant bot.

Reply in a {tone} way. You can use Bangla, English, or mixed Banglish depending on the user's message.
Do not save anything to Notion. Just answer the user.

User message:
{message[:3000]}"""
        providers = []
        if self.groq_key:
            providers.append(self.call_groq)
        if self.deepseek_key:
            providers.append(self.call_deepseek)

        for provider_fn in providers:
            try:
                return provider_fn(prompt, max_tokens=900, temperature=0.7 if playful else 0.2)
            except Exception:
                continue
        return "I could not answer right now. Try again in a moment."

    def read_image(self, image_path_or_url: str, is_url: bool = False) -> str:
        return read_image_with_gemini(image_path_or_url, gemini_key=self.gemini_key, is_url=is_url)

    def transcribe_audio(self, audio_path: str) -> str:
        if not self.groq_key:
            raise RuntimeError("GROQ_API_KEY is not set")
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb > 24:
            trunc = audio_path.replace(".mp3", "_trunc.mp3")
            subprocess.run(
                ["ffmpeg", "-i", audio_path, "-t", "600", "-y", trunc],
                capture_output=True,
                timeout=60,
            )
            audio_path = trunc

        with open(audio_path, "rb") as f:
            data = f.read()

        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {self.groq_key}"},
            files={"file": ("audio.mp3", data, "audio/mpeg")},
            data={"model": "whisper-large-v3", "response_format": "text", "temperature": "0"},
            timeout=120,
        )
        if r.status_code == 200:
            return r.text.strip()
        raise RuntimeError(f"Whisper {r.status_code}: {r.text[:150]}")
