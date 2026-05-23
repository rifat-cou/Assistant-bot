"""
Campus Assistant — Telegram Bot
Handles: Screenshots, Videos (up to 2GB), PDFs, Text
AI: DeepSeek + Groq + Gemini Vision (free, rotating)
Storage: Notion database (same one as Discord bot)
"""

import os
import json
import requests
import base64
import subprocess
import tempfile
import logging
from datetime import datetime

# ─────────────────────────────────────────────────────────────
#  CREDENTIALS — set these in Railway environment variables
# ─────────────────────────────────────────────────────────────
TELEGRAM_TOKEN  = os.environ["TELEGRAM_TOKEN"]       # from @BotFather
GROQ_KEY        = os.environ["GROQ_API_KEY"]          # console.groq.com (free)
DEEPSEEK_KEY    = os.environ["DEEPSEEK_API_KEY"]      # platform.deepseek.com (free)
GEMINI_KEY      = os.environ["GEMINI_API_KEY"]        # aistudio.google.com (free)
NOTION_SECRET   = os.environ["NOTION_SECRET"]         # notion.so/my-integrations
NOTION_DB_ID    = os.environ["NOTION_DB_ID"]          # same DB as Discord bot

from notion_client import Client
notion = Client(auth=NOTION_SECRET)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  MULTI-LLM ROUTER — tries providers in order, falls back
# ─────────────────────────────────────────────────────────────

class LLMRouter:
    """
    Intelligent router that picks the best free LLM for each task.
    Falls back automatically if one hits rate limits.
    
    Strategy:
    - DeepSeek  → heavy categorization, long text (1M tokens/month free)
    - Groq      → fast real-time responses, short prompts
    - Gemini    → image OCR, multimodal tasks
    """

    @staticmethod
    def call_deepseek(prompt: str, max_tokens: int = 700) -> str:
        """DeepSeek — 1M free tokens/month. OpenAI-compatible format."""
        r = requests.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {DEEPSEEK_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1
            },
            timeout=30
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError(f"DeepSeek {r.status_code}: {r.text[:150]}")

    @staticmethod
    def call_groq(prompt: str, max_tokens: int = 700) -> str:
        """Groq — ultra-fast, ~14K requests/day free."""
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
                "temperature": 0.1
            },
            timeout=25
        )
        if r.status_code == 200:
            return r.json()["choices"][0]["message"]["content"].strip()
        raise RuntimeError(f"Groq {r.status_code}: {r.text[:150]}")

    @classmethod
    def categorize(cls, text: str, source: str = "") -> dict:
        """
        Try DeepSeek first (larger quota), fall back to Groq.
        Used for: categorizing screenshots, notes, links, videos.
        """
        prompt = build_categorize_prompt(text, source)

        for provider_name, provider_fn in [
            ("DeepSeek", cls.call_deepseek),
            ("Groq",     cls.call_groq),
        ]:
            try:
                raw = provider_fn(prompt, max_tokens=700)
                raw = raw.replace("```json", "").replace("```", "").strip()
                result = json.loads(raw)
                result["_llm_used"] = provider_name
                return result
            except Exception as e:
                log.warning(f"{provider_name} failed: {e}. Trying next...")

        # Hard fallback — return minimal structure
        return {
            "title": f"Saved content from {source}",
            "category": "Social Post",
            "sub_tags": [source or "saved"],
            "language": "Both",
            "summary_en": text[:200],
            "summary_bn": "বিষয়বস্তু সংরক্ষিত হয়েছে।",
            "relevance_score": 5.0,
            "has_deadline": False,
            "deadline_date": None,
            "action": "Save only",
            "keywords": [],
            "_llm_used": "fallback"
        }

    @staticmethod
    def transcribe_audio(audio_path: str) -> str:
        """Groq Whisper — always use Groq for audio (DeepSeek has no audio endpoint)."""
        file_size_mb = os.path.getsize(audio_path) / (1024 * 1024)
        if file_size_mb > 24:
            # Truncate to first 10 minutes
            trunc = audio_path.replace(".mp3", "_trunc.mp3")
            subprocess.run(
                ["ffmpeg", "-i", audio_path, "-t", "600", "-y", trunc],
                capture_output=True, timeout=60
            )
            audio_path = trunc

        with open(audio_path, "rb") as f:
            data = f.read()

        r = requests.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers={"Authorization": f"Bearer {GROQ_KEY}"},
            files={"file": ("audio.mp3", data, "audio/mpeg")},
            data={"model": "whisper-large-v3", "response_format": "text", "temperature": "0"},
            timeout=120
        )
        if r.status_code == 200:
            return r.text.strip()
        raise RuntimeError(f"Whisper {r.status_code}: {r.text[:150]}")

    @staticmethod
    def read_image(image_path_or_url: str, is_url: bool = False) -> str:
        """Gemini Vision — always use Gemini for images (best free OCR)."""
        if is_url:
            img_data = requests.get(image_path_or_url, timeout=15).content
        else:
            with open(image_path_or_url, "rb") as f:
                img_data = f.read()

        img_b64 = base64.b64encode(img_data).decode("utf-8")

        payload = {
            "contents": [{
                "parts": [
                    {"inlineData": {"mimeType": "image/jpeg", "data": img_b64}},
                    {"text": (
                        "This image may contain Bengali (Bangla) and/or English text. "
                        "Please: 1) Extract ALL text visible exactly as written — preserve Bengali script. "
                        "2) List any URLs, website names, or tool names you see. "
                        "3) Describe what type of content this is (screenshot of website, slide, social post, etc.). "
                        "4) Note the language(s). Be thorough — miss nothing."
                    )}
                ]
            }]
        }
        r = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}",
            json=payload, timeout=30
        )
        if r.status_code == 200:
            return r.json()["candidates"][0]["content"]["parts"][0]["text"]
        raise RuntimeError(f"Gemini Vision {r.status_code}: {r.text[:150]}")


llm = LLMRouter()

# ─────────────────────────────────────────────────────────────
#  CATEGORIZATION PROMPT
# ─────────────────────────────────────────────────────────────

def build_categorize_prompt(text: str, source: str = "") -> str:
    return f"""You are a personal content organizer for Marof, a Bangladeshi student building "AI Campus" — an AI education platform.

Content source type: {source}
Content text (may be in Bangla, English, or both):
{text[:2500]}

Analyze carefully. Return ONLY valid JSON, no markdown, no explanation:
{{
  "title": "clear descriptive title in English, max 12 words",
  "category": "pick exactly one: Scholarship | Research | AI Tools | Tech News | Learning | Career | Video | Image | Social Post | Paper | Template | Idea | News | Website | URL List",
  "sub_tags": ["tag1", "tag2", "tag3"],
  "language": "Bangla | English | Both",
  "summary_en": "2 clear sentences summarizing content in English",
  "summary_bn": "২টি বাক্যে বাংলায় সারসংক্ষেপ (Bengali script)",
  "relevance_score": 8.0,
  "relevance_reason": "one sentence why this is relevant to an AI student in Bangladesh",
  "has_deadline": false,
  "deadline_date": null,
  "action": "Apply | Read | Watch | Save only | Visit",
  "extracted_urls": ["any URLs or website names found in the content"],
  "extracted_tools": ["any tool names, app names, or software mentioned"],
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4"]
}}"""


# ─────────────────────────────────────────────────────────────
#  NOTION SAVE
# ─────────────────────────────────────────────────────────────

def save_to_notion(ai_data: dict, source_url: str, raw_text: str, file_type: str = "") -> str:
    """Save to Notion. Returns Notion page URL."""
    tags = [{"name": t} for t in ai_data.get("sub_tags", [])[:10]]

    # Add extracted URLs as extra tags
    for url in ai_data.get("extracted_urls", [])[:3]:
        if url and len(url) < 50:
            tags.append({"name": f"url:{url[:40]}"})

    props = {
        "Title":        {"title": [{"text": {"content": ai_data.get("title", "Untitled")[:100]}}]},
        "Category":     {"select": {"name": ai_data.get("category", "Social Post")}},
        "Tags":         {"multi_select": tags},
        "Language":     {"select": {"name": ai_data.get("language", "Both")}},
        "Summary EN":   {"rich_text": [{"text": {"content": ai_data.get("summary_en", "")[:2000]}}]},
        "Summary BN":   {"rich_text": [{"text": {"content": ai_data.get("summary_bn", "")[:2000]}}]},
        "Relevance":    {"number": float(ai_data.get("relevance_score", 5.0))},
        "Action":       {"select": {"name": ai_data.get("action", "Save only")}},
        "Has Deadline": {"checkbox": bool(ai_data.get("has_deadline", False))},
        "Raw Text":     {"rich_text": [{"text": {"content": raw_text[:2000]}}]},
        "Saved At":     {"date": {"start": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")}},
    }
    if source_url:
        props["URL"] = {"url": source_url}
    if ai_data.get("deadline_date"):
        props["Deadline"] = {"date": {"start": ai_data["deadline_date"]}}

    page = notion.pages.create(parent={"database_id": NOTION_DB_ID}, properties=props)
    return page.get("url", "")


# ─────────────────────────────────────────────────────────────
#  VIDEO PROCESSING
# ─────────────────────────────────────────────────────────────

def process_video_file(file_path: str) -> dict:
    """Extract audio, transcribe with Whisper, describe thumbnail with Gemini."""
    results = {"transcript": "", "visual_desc": "", "duration": "unknown"}

    # Duration
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True, text=True, timeout=15
        )
        secs = float(r.stdout.strip())
        results["duration"] = f"{int(secs//60)}:{int(secs%60):02d}"
    except Exception:
        pass

    # Extract audio
    audio_path = file_path + "_audio.mp3"
    try:
        subprocess.run(
            ["ffmpeg", "-i", file_path, "-vn", "-acodec", "mp3",
             "-ar", "16000", "-ac", "1", "-ab", "64k", "-y", audio_path],
            capture_output=True, timeout=120
        )
        if os.path.exists(audio_path):
            results["transcript"] = llm.transcribe_audio(audio_path)
    except Exception as e:
        results["transcript"] = f"[Transcription failed: {e}]"
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

    # Extract thumbnail
    thumb_path = file_path + "_thumb.jpg"
    try:
        subprocess.run(
            ["ffmpeg", "-i", file_path, "-ss", "00:00:01",
             "-vframes", "1", "-q:v", "2", "-y", thumb_path],
            capture_output=True, timeout=30
        )
        if os.path.exists(thumb_path):
            results["visual_desc"] = llm.read_image(thumb_path)
    except Exception as e:
        results["visual_desc"] = f"[Visual description failed: {e}]"
    finally:
        if os.path.exists(thumb_path):
            os.unlink(thumb_path)

    return results


# ─────────────────────────────────────────────────────────────
#  TELEGRAM API HELPERS
# ─────────────────────────────────────────────────────────────

BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}"
FILE_URL = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}"

def tg_send(chat_id: int, text: str, parse_mode: str = "Markdown"):
    """Send a message to Telegram."""
    requests.post(f"{BASE_URL}/sendMessage", json={
        "chat_id": chat_id,
        "text": text,
        "parse_mode": parse_mode,
        "disable_web_page_preview": True
    }, timeout=15)

def tg_download_file(file_id: str, suffix: str) -> str:
    """Download a file from Telegram servers to a temp file."""
    # Step 1: Get file path from Telegram
    r = requests.get(f"{BASE_URL}/getFile", params={"file_id": file_id}, timeout=15)
    file_path = r.json()["result"]["file_path"]

    # Step 2: Download the actual file
    file_url = f"{FILE_URL}/{file_path}"
    response  = requests.get(file_url, timeout=120, stream=True)

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    for chunk in response.iter_content(chunk_size=8192):
        tmp.write(chunk)
    tmp.close()
    return tmp.name

def format_notion_reply(ai_data: dict, notion_url: str, file_type: str = "") -> str:
    """Build the Telegram reply message with organized content."""
    provider = ai_data.get("_llm_used", "AI")
    score    = ai_data.get("relevance_score", "?")
    tags     = " ".join([f"`{t}`" for t in ai_data.get("sub_tags", [])[:5]])
    urls     = ai_data.get("extracted_urls", [])
    tools    = ai_data.get("extracted_tools", [])
    deadline = f"\n📅 *Deadline:* `{ai_data['deadline_date']}`" if ai_data.get("deadline_date") else ""

    url_section = ""
    if urls:
        url_list = "\n".join([f"  • {u}" for u in urls[:5]])
        url_section = f"\n\n🔗 *URLs found:*\n{url_list}"

    tool_section = ""
    if tools:
        tool_list = ", ".join(tools[:6])
        tool_section = f"\n\n🛠 *Tools mentioned:* {tool_list}"

    return f"""✅ *Saved & organized*

📌 *{ai_data.get('title', 'Untitled')}*

📝 *Summary (EN):*
{ai_data.get('summary_en', '—')}

🇧🇩 *সারসংক্ষেপ:*
{ai_data.get('summary_bn', '—')}

🏷 *Category:* `{ai_data.get('category', '?')}`
🌐 *Language:* `{ai_data.get('language', '?')}`
⭐ *Relevance:* `{score}/10`
🎯 *Action:* `{ai_data.get('action', 'Save only')}`{deadline}
🏷 *Tags:* {tags}{url_section}{tool_section}

📒 [Open in Notion]({notion_url})
🤖 _{provider}_"""


# ─────────────────────────────────────────────────────────────
#  MESSAGE HANDLER — core logic
# ─────────────────────────────────────────────────────────────

def handle_update(update: dict):
    """Process one incoming Telegram update."""
    msg = update.get("message") or update.get("channel_post")
    if not msg:
        return

    chat_id = msg["chat"]["id"]
    tmp_path = None

    try:
        # ── TEXT MESSAGE ──────────────────────────────────────
        if "text" in msg:
            text = msg["text"].strip()

            # Commands
            if text == "/start":
                tg_send(chat_id, (
                    "👋 *Campus Assistant* is ready\\!\n\n"
                    "Just send me:\n"
                    "• 📸 *Screenshot* → I read all text with AI Vision\n"
                    "• 🎬 *Video* → I transcribe speech \\(Bangla \\+ English\\)\n"
                    "• 🔗 *Link* → I fetch and organize it\n"
                    "• 📝 *Copied text* → I categorize and save it\n"
                    "• 📄 *PDF* → I extract and organize content\n\n"
                    "_No commands needed\\. Just send the file\\._"
                ), parse_mode="MarkdownV2")
                return

            if text == "/stats":
                results = notion.databases.query(database_id=NOTION_DB_ID, page_size=100)["results"]
                counts = {}
                for r in results:
                    p   = r["properties"]
                    cat = p["Category"]["select"]["name"] if p.get("Category") and p["Category"].get("select") else "Other"
                    counts[cat] = counts.get(cat, 0) + 1
                lines = [f"• {cat}: {n}" for cat, n in sorted(counts.items(), key=lambda x: -x[1])]
                tg_send(chat_id, f"📊 *Your library — {len(results)} items*\n\n" + "\n".join(lines))
                return

            if text.startswith("/find "):
                query = text[6:].strip()
                results = notion.databases.query(
                    database_id=NOTION_DB_ID,
                    filter={"property": "Title", "title": {"contains": query}},
                    sorts=[{"property": "Saved At", "direction": "descending"}],
                    page_size=5
                )["results"]
                if not results:
                    tg_send(chat_id, f"Nothing found for '{query}'.")
                    return
                lines = []
                for r in results[:5]:
                    p     = r["properties"]
                    title = p["Title"]["title"][0]["plain_text"] if p["Title"]["title"] else "Untitled"
                    cat   = p["Category"]["select"]["name"] if p.get("Category") and p["Category"].get("select") else "?"
                    url   = p.get("URL", {}).get("url") or ""
                    link  = f"[{title}]({url})" if url else title
                    lines.append(f"• `{cat}` — {link}")
                tg_send(chat_id, f"🔍 *Results for '{query}':*\n\n" + "\n".join(lines))
                return

            if text.startswith("/list "):
                category = text[6:].strip().title()
                results = notion.databases.query(
                    database_id=NOTION_DB_ID,
                    filter={"property": "Category", "select": {"equals": category}},
                    sorts=[{"property": "Saved At", "direction": "descending"}],
                    page_size=10
                )["results"]
                if not results:
                    tg_send(chat_id, f"Nothing in '{category}' yet.")
                    return
                lines = []
                for r in results[:10]:
                    p     = r["properties"]
                    title = p["Title"]["title"][0]["plain_text"] if p["Title"]["title"] else "Untitled"
                    url   = p.get("URL", {}).get("url") or ""
                    lines.append(f"• [{title}]({url})" if url else f"• {title}")
                tg_send(chat_id, f"📂 *{category} — {len(results)} saved:*\n\n" + "\n".join(lines))
                return

            if text == "/deadlines":
                results = notion.databases.query(
                    database_id=NOTION_DB_ID,
                    filter={"property": "Has Deadline", "checkbox": {"equals": True}},
                    sorts=[{"property": "Deadline", "direction": "ascending"}],
                    page_size=8
                )["results"]
                if not results:
                    tg_send(chat_id, "No deadline items saved yet.")
                    return
                lines = []
                for r in results:
                    p     = r["properties"]
                    title = p["Title"]["title"][0]["plain_text"] if p["Title"]["title"] else "Untitled"
                    dl    = p["Deadline"]["date"]["start"] if p.get("Deadline") and p["Deadline"].get("date") else "?"
                    lines.append(f"• `{dl}` — {title}")
                tg_send(chat_id, "📅 *Upcoming deadlines:*\n\n" + "\n".join(lines))
                return

            # Plain text or URL — save it
            tg_send(chat_id, "⏳ Reading and organizing...")
            source_type = "url" if text.startswith("http") else "text"
            ai_data  = llm.categorize(text, source=source_type)
            notion_url = save_to_notion(ai_data, text if source_type == "url" else "", text)
            tg_send(chat_id, format_notion_reply(ai_data, notion_url, source_type))
            return

        # ── PHOTO / SCREENSHOT ────────────────────────────────
        if "photo" in msg:
            tg_send(chat_id, "⏳ Reading image text with AI Vision...")
            # Telegram sends multiple sizes — pick largest
            photo    = msg["photo"][-1]
            tmp_path = tg_download_file(photo["file_id"], suffix=".jpg")
            extracted_text = llm.read_image(tmp_path)
            ai_data  = llm.categorize(extracted_text, source="screenshot")
            notion_url = save_to_notion(ai_data, "", extracted_text)
            tg_send(chat_id, format_notion_reply(ai_data, notion_url, "image"))
            return

        # ── DOCUMENT (could be image, video, PDF sent as file) ─
        if "document" in msg:
            doc      = msg["document"]
            filename = doc.get("file_name", "file")
            mime     = doc.get("mime_type", "")
            ext      = os.path.splitext(filename)[1].lower()

            # IMAGE sent as document (uncompressed)
            if mime.startswith("image/") or ext in {".png", ".jpg", ".jpeg", ".webp", ".heic", ".bmp"}:
                tg_send(chat_id, "⏳ Reading image text with AI Vision...")
                tmp_path = tg_download_file(doc["file_id"], suffix=ext or ".jpg")
                extracted_text = llm.read_image(tmp_path)
                ai_data  = llm.categorize(extracted_text, source="screenshot")
                notion_url = save_to_notion(ai_data, "", extracted_text)
                tg_send(chat_id, format_notion_reply(ai_data, notion_url, "image"))
                return

            # PDF
            if mime == "application/pdf" or ext == ".pdf":
                tg_send(chat_id, "⏳ Processing PDF...")
                tmp_path = tg_download_file(doc["file_id"], suffix=".pdf")
                # Extract text from PDF using pdftotext (installed via nixpacks)
                try:
                    r = subprocess.run(
                        ["pdftotext", "-layout", tmp_path, "-"],
                        capture_output=True, text=True, timeout=60
                    )
                    pdf_text = r.stdout.strip()[:3000] or f"PDF file: {filename}"
                except Exception:
                    pdf_text = f"PDF file: {filename}"
                ai_data  = llm.categorize(pdf_text, source="pdf")
                notion_url = save_to_notion(ai_data, "", pdf_text)
                tg_send(chat_id, format_notion_reply(ai_data, notion_url, "pdf"))
                return

            # VIDEO sent as document
            if mime.startswith("video/") or ext in {".mp4", ".mov", ".avi", ".webm", ".mkv", ".m4v"}:
                tg_send(chat_id, f"🎬 Processing video `{filename}`...\nExtracting audio → transcribing → analyzing\n⏱ Takes 30–90 seconds for longer videos.")
                tmp_path = tg_download_file(doc["file_id"], suffix=ext or ".mp4")
                video_data = process_video_file(tmp_path)
                combined = f"Video: {filename}\nDuration: {video_data['duration']}\n\nVisual: {video_data['visual_desc']}\n\nTranscript:\n{video_data['transcript']}"
                ai_data  = llm.categorize(combined, source="video")
                notion_url = save_to_notion(ai_data, "", combined)
                reply    = format_notion_reply(ai_data, notion_url, "video")
                if video_data.get("transcript"):
                    reply += f"\n\n🎙 *Transcript preview:*\n_{video_data['transcript'][:300]}..._"
                tg_send(chat_id, reply)
                return

        # ── VIDEO (compressed, sent as video message) ─────────
        if "video" in msg:
            vid      = msg["video"]
            filename = vid.get("file_name", "video.mp4")
            tg_send(chat_id, f"🎬 Processing video...\nExtracting speech → transcribing (Bangla + English)\n⏱ Takes 30–90 seconds.")
            tmp_path = tg_download_file(vid["file_id"], suffix=".mp4")
            video_data = process_video_file(tmp_path)
            combined = f"Video: {filename}\nDuration: {video_data['duration']}\n\nVisual: {video_data['visual_desc']}\n\nTranscript:\n{video_data['transcript']}"
            ai_data  = llm.categorize(combined, source="video")
            notion_url = save_to_notion(ai_data, "", combined)
            reply    = format_notion_reply(ai_data, notion_url, "video")
            if video_data.get("transcript"):
                reply += f"\n\n🎙 *Transcript preview:*\n_{video_data['transcript'][:300]}..._"
            tg_send(chat_id, reply)
            return

        # ── VOICE MESSAGE ─────────────────────────────────────
        if "voice" in msg:
            tg_send(chat_id, "🎙 Transcribing voice message...")
            tmp_path = tg_download_file(msg["voice"]["file_id"], suffix=".ogg")
            # Convert ogg to mp3 first
            mp3_path = tmp_path.replace(".ogg", ".mp3")
            subprocess.run(["ffmpeg", "-i", tmp_path, "-y", mp3_path], capture_output=True, timeout=30)
            transcript = llm.transcribe_audio(mp3_path)
            ai_data  = llm.categorize(transcript, source="voice")
            notion_url = save_to_notion(ai_data, "", transcript)
            tg_send(chat_id, format_notion_reply(ai_data, notion_url, "voice"))
            if os.path.exists(mp3_path):
                os.unlink(mp3_path)
            return

    except Exception as e:
        log.error(f"Handler error: {e}", exc_info=True)
        tg_send(chat_id, f"❌ Error processing content: `{str(e)[:200]}`\n\nTry again or send the text directly using plain text message.")

    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# ─────────────────────────────────────────────────────────────
#  POLLING LOOP
# ─────────────────────────────────────────────────────────────

def run_bot():
    """Long-polling loop — checks for new messages every 2 seconds."""
    log.info("Campus Assistant Telegram Bot starting...")
    offset = 0

    while True:
        try:
            r = requests.get(
                f"{BASE_URL}/getUpdates",
                params={"offset": offset, "timeout": 30, "allowed_updates": ["message"]},
                timeout=35
            )
            updates = r.json().get("result", [])
            for update in updates:
                offset = update["update_id"] + 1
                try:
                    handle_update(update)
                except Exception as e:
                    log.error(f"Update error: {e}")
        except requests.exceptions.ConnectionError:
            import time
            log.warning("Connection lost. Retrying in 5s...")
            time.sleep(5)
        except Exception as e:
            import time
            log.error(f"Polling error: {e}")
            time.sleep(2)


if __name__ == "__main__":
    run_bot()
