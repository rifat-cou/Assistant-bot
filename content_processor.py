import os
import re
import subprocess

import requests


def detect_content_type(text: str) -> str:
    t = text.lower().strip()
    if any(x in t for x in ["youtube.com/watch", "youtu.be/", "youtube.com/shorts/", "youtube.com/live/", "m.youtube.com"]):
        return "youtube"
    if any(x in t for x in ["facebook.com", "fb.com", "fb.watch", "m.facebook.com"]):
        return "facebook"
    if any(x in t for x in ["instagram.com", "instagr.am"]):
        return "instagram"
    if any(x in t for x in ["tiktok.com", "vm.tiktok.com"]):
        return "tiktok"
    if any(x in t for x in ["twitter.com", "x.com", "t.co"]):
        return "twitter"
    if text.startswith("http://") or text.startswith("https://"):
        return "url"
    return "text"


def fetch_youtube(url: str) -> dict:
    try:
        r = requests.get(f"https://www.youtube.com/oembed?url={url}&format=json", timeout=10)
        data = r.json()
        title = data.get("title", "YouTube Video")
        channel = data.get("author_name", "Unknown Channel")
        vid_id = ""
        for pattern in [r"v=([^&]+)", r"youtu\.be/([^?]+)", r"shorts/([^?/]+)"]:
            m = re.search(pattern, url)
            if m:
                vid_id = m.group(1)
                break
        description = fetch_via_jina(url)
        return {
            "title": title,
            "channel": channel,
            "video_id": vid_id,
            "text": f"YouTube video: {title} by {channel}. {description[:500]}",
            "thumbnail": f"https://img.youtube.com/vi/{vid_id}/hqdefault.jpg" if vid_id else "",
        }
    except Exception as e:
        return {"title": "YouTube Video", "text": f"YouTube link: {url}", "error": str(e)}


def fetch_facebook(url: str) -> dict:
    jina_text = fetch_via_jina(url)
    if jina_text and len(jina_text) > 100 and "log in" not in jina_text.lower():
        return {"title": "Facebook Post", "text": jina_text[:1500], "source": "facebook", "warning": None}

    og_data = fetch_open_graph(url)
    if og_data.get("title") or og_data.get("description"):
        return {
            "title": og_data.get("title", "Facebook Post"),
            "text": f"{og_data.get('title', '')} - {og_data.get('description', '')}",
            "source": "facebook",
            "warning": "partial",
        }

    return {
        "title": "Facebook Post",
        "text": f"Facebook link saved: {url}",
        "source": "facebook",
        "warning": "blocked",
    }


def fetch_instagram(url: str) -> dict:
    jina_text = fetch_via_jina(url)
    og_data = fetch_open_graph(url)
    text = jina_text if (jina_text and len(jina_text) > 80) else f"{og_data.get('title','')} {og_data.get('description','')}".strip()
    return {"title": og_data.get("title", "Instagram Post"), "text": text or f"Instagram link: {url}", "source": "instagram"}


def fetch_tiktok(url: str) -> dict:
    try:
        r = requests.get(f"https://www.tiktok.com/oembed?url={url}", timeout=10, headers={"User-Agent": "Mozilla/5.0"})
        data = r.json()
        title = data.get("title", "TikTok Video")
        author = data.get("author_name", "")
        return {"title": title, "text": f"TikTok video by @{author}: {title}", "source": "tiktok"}
    except Exception:
        return {"title": "TikTok Video", "text": f"TikTok link: {url}", "source": "tiktok"}


def fetch_twitter(url: str) -> dict:
    jina_text = fetch_via_jina(url)
    og = fetch_open_graph(url)
    text = jina_text if (jina_text and len(jina_text) > 50) else f"{og.get('title','')} {og.get('description','')}".strip()
    return {"title": og.get("title", "Tweet / X Post"), "text": text or f"Twitter/X link: {url}", "source": "twitter"}


def fetch_generic_url(url: str) -> dict:
    jina_text = fetch_via_jina(url)
    og = fetch_open_graph(url)
    return {
        "title": og.get("title", "Web Article"),
        "text": jina_text[:2000] if jina_text else f"{og.get('title','')} {og.get('description','')}",
        "source": "web",
    }


def fetch_via_jina(url: str) -> str:
    try:
        r = requests.get(
            f"https://r.jina.ai/{url}",
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; CampusAssistant/1.0)", "Accept": "text/plain"},
        )
        text = r.text.strip()
        lines = [line for line in text.split("\n") if len(line.strip()) > 30]
        return "\n".join(lines[:40])
    except Exception:
        return ""


def fetch_open_graph(url: str) -> dict:
    try:
        r = requests.get(url, timeout=10, headers={"User-Agent": "Mozilla/5.0 (compatible; facebookexternalhit/1.1)"})
        html = r.text

        def get_meta(prop):
            patterns = [
                f'property="{prop}" content="([^"]*)"',
                f"property='{prop}' content='([^']*)'",
                f'name="{prop}" content="([^"]*)"',
                f'content="([^"]*)" property="{prop}"',
            ]
            for pattern in patterns:
                match = re.search(pattern, html, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
            return ""

        return {
            "title": get_meta("og:title") or get_meta("twitter:title"),
            "description": get_meta("og:description") or get_meta("twitter:description"),
            "image": get_meta("og:image"),
            "site_name": get_meta("og:site_name"),
        }
    except Exception:
        return {}


def fetch_readable_content(content: str, source_type: str = None, llm=None) -> tuple:
    source_type = source_type or detect_content_type(content)
    url = content if content.startswith("http") else ""
    warning = None

    if source_type == "youtube":
        fetched = fetch_youtube(url)
    elif source_type == "facebook":
        fetched = fetch_facebook(url)
        warning = fetched.get("warning")
    elif source_type == "instagram":
        fetched = fetch_instagram(url)
    elif source_type == "tiktok":
        fetched = fetch_tiktok(url)
    elif source_type == "twitter":
        fetched = fetch_twitter(url)
    elif source_type == "url":
        fetched = fetch_generic_url(url)
    elif source_type == "image" and llm:
        text = llm.read_image(content, is_url=True)
        fetched = {"title": "Image / Screenshot", "text": text}
    else:
        fetched = {"title": "Text Note", "text": content}

    return source_type, url, fetched.get("text", content), fetched, warning


def process_video_file(file_path: str, llm) -> dict:
    results = {"transcript": "", "visual_desc": "", "duration": "unknown"}
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", file_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
        secs = float(r.stdout.strip())
        results["duration"] = f"{int(secs // 60)}:{int(secs % 60):02d}"
    except Exception:
        pass

    audio_path = file_path + "_audio.mp3"
    try:
        subprocess.run(
            ["ffmpeg", "-i", file_path, "-vn", "-acodec", "mp3", "-ar", "16000", "-ac", "1", "-ab", "64k", "-y", audio_path],
            capture_output=True,
            timeout=120,
        )
        if os.path.exists(audio_path):
            results["transcript"] = llm.transcribe_audio(audio_path)
    except Exception as e:
        results["transcript"] = f"[Transcription failed: {e}]"
    finally:
        if os.path.exists(audio_path):
            os.unlink(audio_path)

    thumb_path = file_path + "_thumb.jpg"
    try:
        subprocess.run(
            ["ffmpeg", "-i", file_path, "-ss", "00:00:01", "-vframes", "1", "-q:v", "2", "-y", thumb_path],
            capture_output=True,
            timeout=30,
        )
        if os.path.exists(thumb_path):
            results["visual_desc"] = llm.read_image(thumb_path)
    except Exception as e:
        results["visual_desc"] = f"[Visual description failed: {e}]"
    finally:
        if os.path.exists(thumb_path):
            os.unlink(thumb_path)

    return results
