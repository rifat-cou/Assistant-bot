"""
Campus Assistant Discord Bot — FIXED VERSION
Handles: Facebook, YouTube, Instagram, TikTok, images (OCR), plain text, any URL
Uses: Groq (AI), Gemini Vision (image OCR), Jina AI (URL reader), YouTube oEmbed (free)
"""

import discord
from discord import app_commands
import requests
import json
import os
import re
from notion_client import Client
from datetime import datetime
from assistant_core import (
    CATEGORIES,
    build_categorize_prompt,
    clean_json_response,
    normalize_ai_data,
    normalize_category,
)

# ─────────────────────────────────────────────────────────────
#  CREDENTIALS  (set these in Railway environment variables)
# ─────────────────────────────────────────────────────────────
DISCORD_TOKEN  = os.environ["DISCORD_TOKEN"]
GROQ_KEY       = os.environ["GROQ_API_KEY"]
GEMINI_KEY     = os.environ["GEMINI_API_KEY"]      # for image OCR — free at aistudio.google.com
NOTION_SECRET  = os.environ["NOTION_SECRET"]
NOTION_DB_ID   = os.environ["NOTION_DB_ID"]

notion = Client(auth=NOTION_SECRET)
_notion_property_cache = None

intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)


def notion_properties() -> set:
    global _notion_property_cache
    if _notion_property_cache is None:
        try:
            db = notion.databases.retrieve(database_id=NOTION_DB_ID)
            _notion_property_cache = set(db.get("properties", {}).keys())
        except Exception:
            _notion_property_cache = set()
    return _notion_property_cache


def add_prop_if_exists(props: dict, name: str, value: dict):
    if name in notion_properties():
        props[name] = value


# ═════════════════════════════════════════════════════════════
#  STEP 1 — DETECT WHAT TYPE OF CONTENT WAS SHARED
# ═════════════════════════════════════════════════════════════

def detect_content_type(text: str) -> str:
    """
    Figures out what type of content the user shared.
    Returns one of: youtube | facebook | instagram | tiktok | twitter | url | text
    """
    t = text.lower().strip()

    # YouTube — many URL formats
    if any(x in t for x in [
        "youtube.com/watch", "youtu.be/", "youtube.com/shorts/",
        "youtube.com/live/", "m.youtube.com"
    ]):
        return "youtube"

    # Facebook — posts, reels, photos, videos, stories
    if any(x in t for x in [
        "facebook.com", "fb.com", "fb.watch", "m.facebook.com"
    ]):
        return "facebook"

    # Instagram
    if any(x in t for x in ["instagram.com", "instagr.am"]):
        return "instagram"

    # TikTok
    if any(x in t for x in ["tiktok.com", "vm.tiktok.com"]):
        return "tiktok"

    # Twitter / X
    if any(x in t for x in ["twitter.com", "x.com", "t.co"]):
        return "twitter"

    # Any other URL
    if text.startswith("http://") or text.startswith("https://"):
        return "url"

    # Plain text (copied post, note)
    return "text"


# ═════════════════════════════════════════════════════════════
#  STEP 2 — FETCH READABLE CONTENT FROM EACH TYPE
# ═════════════════════════════════════════════════════════════

def fetch_youtube(url: str) -> dict:
    """
    YouTube: use the free oEmbed API — no API key needed.
    Gets title, author, thumbnail. Then Jina AI for description.
    """
    try:
        # oEmbed gives title and channel — completely free, no key
        oembed_url = f"https://www.youtube.com/oembed?url={url}&format=json"
        r = requests.get(oembed_url, timeout=10)
        data = r.json()
        title   = data.get("title", "YouTube Video")
        channel = data.get("author_name", "Unknown Channel")

        # Extract video ID for thumbnail
        vid_id = ""
        for pattern in [r"v=([^&]+)", r"youtu\.be/([^?]+)", r"shorts/([^?/]+)"]:
            m = re.search(pattern, url)
            if m:
                vid_id = m.group(1)
                break

        # Try Jina AI to get description text
        description = fetch_via_jina(url)

        return {
            "title": title,
            "channel": channel,
            "video_id": vid_id,
            "text": f"YouTube video: {title} by {channel}. {description[:500]}",
            "thumbnail": f"https://img.youtube.com/vi/{vid_id}/hqdefault.jpg" if vid_id else ""
        }
    except Exception as e:
        return {"title": "YouTube Video", "text": f"YouTube link: {url}", "error": str(e)}


def fetch_facebook(url: str) -> dict:
    """
    Facebook BLOCKS all bots — cannot read content from URL directly.
    Strategy: use Jina AI reader (sometimes works for public posts),
    then fall back to Open Graph metadata only.
    """
    # Try Jina AI first (works for some public FB pages)
    jina_text = fetch_via_jina(url)

    if jina_text and len(jina_text) > 100 and "log in" not in jina_text.lower():
        return {
            "title": "Facebook Post",
            "text": jina_text[:1500],
            "source": "facebook",
            "warning": None
        }

    # Try Open Graph metadata
    og_data = fetch_open_graph(url)
    if og_data.get("title") or og_data.get("description"):
        return {
            "title": og_data.get("title", "Facebook Post"),
            "text": f"{og_data.get('title', '')} — {og_data.get('description', '')}",
            "source": "facebook",
            "warning": "partial"  # only metadata, not full post
        }

    # Facebook fully blocked this — tell user to copy text
    return {
        "title": "Facebook Post",
        "text": f"Facebook link saved: {url}",
        "source": "facebook",
        "warning": "blocked"
    }


def fetch_instagram(url: str) -> dict:
    """Instagram also blocks bots. Same strategy as Facebook."""
    jina_text = fetch_via_jina(url)
    og_data   = fetch_open_graph(url)

    text = jina_text if (jina_text and len(jina_text) > 80) else \
           f"{og_data.get('title','')} {og_data.get('description','')}".strip()

    return {
        "title": og_data.get("title", "Instagram Post"),
        "text":  text or f"Instagram link: {url}",
        "source": "instagram"
    }


def fetch_tiktok(url: str) -> dict:
    """TikTok: try oEmbed for title, then Jina for description."""
    try:
        r = requests.get(
            f"https://www.tiktok.com/oembed?url={url}",
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        data = r.json()
        title  = data.get("title", "TikTok Video")
        author = data.get("author_name", "")
        return {
            "title": title,
            "text": f"TikTok video by @{author}: {title}",
            "source": "tiktok"
        }
    except Exception:
        return {"title": "TikTok Video", "text": f"TikTok link: {url}", "source": "tiktok"}


def fetch_twitter(url: str) -> dict:
    """Twitter/X: try Jina AI reader."""
    jina_text = fetch_via_jina(url)
    og        = fetch_open_graph(url)
    text = jina_text if (jina_text and len(jina_text) > 50) else \
           f"{og.get('title','')} {og.get('description','')}".strip()
    return {
        "title": og.get("title", "Tweet / X Post"),
        "text":  text or f"Twitter/X link: {url}",
        "source": "twitter"
    }


def fetch_generic_url(url: str) -> dict:
    """For any other URL: Jina AI reader (converts any page to clean text)."""
    jina_text = fetch_via_jina(url)
    og        = fetch_open_graph(url)
    return {
        "title": og.get("title", "Web Article"),
        "text":  jina_text[:2000] if jina_text else f"{og.get('title','')} {og.get('description','')}",
        "source": "web"
    }


# ═════════════════════════════════════════════════════════════
#  HELPER FETCHERS
# ═════════════════════════════════════════════════════════════

def fetch_via_jina(url: str) -> str:
    """
    Jina AI Reader — converts any public webpage into clean readable text.
    Free. No API key needed. URL format: https://r.jina.ai/{target_url}
    Works for: news articles, research papers, LinkedIn posts, some FB pages.
    """
    try:
        jina_url = f"https://r.jina.ai/{url}"
        r = requests.get(
            jina_url,
            timeout=15,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; CampusAssistant/1.0)",
                "Accept": "text/plain"
            }
        )
        text = r.text.strip()
        # Remove junk lines (navigation, cookie banners etc.)
        lines = [l for l in text.split("\n") if len(l.strip()) > 30]
        return "\n".join(lines[:40])  # first 40 meaningful lines
    except Exception:
        return ""


def fetch_open_graph(url: str) -> dict:
    """
    Fetch Open Graph metadata (og:title, og:description) from any URL.
    Works even when full page is blocked — og tags load before login walls.
    """
    try:
        r = requests.get(
            url, timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (compatible; facebookexternalhit/1.1)"}
        )
        html = r.text

        def get_meta(prop):
            patterns = [
                f'property="{prop}" content="([^"]*)"',
                f'property=\'{prop}\' content=\'([^\']*)\'',
                f'name="{prop}" content="([^"]*)"',
                f'content="([^"]*)" property="{prop}"',
            ]
            for p in patterns:
                m = re.search(p, html, re.IGNORECASE)
                if m:
                    return m.group(1).strip()
            return ""

        return {
            "title":       get_meta("og:title") or get_meta("twitter:title"),
            "description": get_meta("og:description") or get_meta("twitter:description"),
            "image":       get_meta("og:image"),
            "site_name":   get_meta("og:site_name"),
        }
    except Exception:
        return {}


def extract_text_from_image_gemini(image_url: str) -> str:
    """
    Use Google Gemini Vision API to read text from an image.
    Free tier: 1500 requests/day.
    Works for: Bengali photocards, English infographics, screenshots of posts.
    """
    try:
        # Download image and encode as base64
        import base64
        img_response = requests.get(image_url, timeout=15)
        img_b64 = base64.b64encode(img_response.content).decode("utf-8")

        # Detect MIME type
        content_type = img_response.headers.get("Content-Type", "image/jpeg")
        if "png" in content_type:
            mime = "image/png"
        elif "webp" in content_type:
            mime = "image/webp"
        elif "gif" in content_type:
            mime = "image/gif"
        else:
            mime = "image/jpeg"

        # Call Gemini Vision
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_KEY}"
        payload = {
            "contents": [{
                "parts": [
                    {
                        "inlineData": {
                            "mimeType": mime,
                            "data": img_b64
                        }
                    },
                    {
                        "text": "This image may contain Bengali (Bangla) or English text, or both. Please: 1) Extract ALL text visible in the image exactly as written. 2) Describe what the image shows (photo, infographic, screenshot, etc.). 3) Note the language(s) used. Return everything you see."
                    }
                ]
            }]
        }
        r = requests.post(api_url, json=payload, timeout=30)
        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception as e:
        return f"Could not read image text: {str(e)}"


# ═════════════════════════════════════════════════════════════
#  STEP 3 — AI CATEGORIZATION (same for all content types)
# ═════════════════════════════════════════════════════════════

def categorize_with_ai(content_text: str, url: str = "", source_type: str = "") -> dict:
    """
    Send extracted content to Groq AI for categorization.
    Returns structured JSON with category, summaries, tags, deadline, etc.
    """
    prompt = build_categorize_prompt(content_text, source=source_type, url=url)

    try:
        r = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 700,
                "temperature": 0.2
            },
            timeout=25
        )
        raw = r.json()["choices"][0]["message"]["content"].strip()
        return normalize_ai_data(clean_json_response(raw), source=source_type, text=content_text)
    except json.JSONDecodeError:
        # AI returned something that's not valid JSON — build a fallback
        return normalize_ai_data({
            "title": f"Saved content from {source_type}",
            "category": "Unknown",
            "sub_tags": [source_type],
            "language": "Both",
            "summary_en": content_text[:200],
            "summary_bn": "বিষয়বস্তু সংরক্ষিত হয়েছে।",
            "relevance_score": 5.0,
            "relevance_reason": "Saved for later review",
            "has_deadline": False,
            "deadline_date": None,
            "action": "Read",
            "keywords": [source_type]
        }, source=source_type, text=content_text)
    except Exception as e:
        raise RuntimeError(f"AI categorization failed: {str(e)}")


def answer_with_ai(question: str, playful: bool = False) -> str:
    """Answer without saving anything to Notion."""
    tone = "friendly, playful, casual" if playful else "helpful, concise, practical"
    prompt = f"""You are Marof's personal assistant bot.

Reply in a {tone} way. You can use Bangla, English, or mixed Banglish depending on the user.
Do not save anything to Notion. Just answer.

User message:
{question[:3000]}"""
    r = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {GROQ_KEY}", "Content-Type": "application/json"},
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 900,
            "temperature": 0.7 if playful else 0.2
        },
        timeout=25
    )
    return r.json()["choices"][0]["message"]["content"].strip()


# ═════════════════════════════════════════════════════════════
#  STEP 4 — SAVE TO NOTION
# ═════════════════════════════════════════════════════════════

def save_to_notion(ai_data: dict, url: str, raw_text: str) -> str:
    """Save categorized content to Notion database. Returns the Notion page URL."""
    tags = [{"name": t} for t in ai_data.get("sub_tags", [])]
    props = {
        "Title":       {"title": [{"text": {"content": ai_data.get("title", "Untitled")[:100]}}]},
        "Category":    {"select": {"name": ai_data.get("category", "Social Post")}},
        "Tags":        {"multi_select": tags},
        "Language":    {"select": {"name": ai_data.get("language", "Both")}},
        "Summary EN":  {"rich_text": [{"text": {"content": ai_data.get("summary_en", "")[:2000]}}]},
        "Summary BN":  {"rich_text": [{"text": {"content": ai_data.get("summary_bn", "")[:2000]}}]},
        "Relevance":   {"number": float(ai_data.get("relevance_score", 5.0))},
        "Action":      {"select": {"name": ai_data.get("action", "Save only")}},
        "Has Deadline":{"checkbox": bool(ai_data.get("has_deadline", False))},
        "Raw Text":    {"rich_text": [{"text": {"content": raw_text[:2000]}}]},
        "Saved At":    {"date": {"start": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")}},
    }
    add_prop_if_exists(props, "Content Type", {"select": {"name": ai_data.get("content_type", "Unknown")}})
    add_prop_if_exists(props, "Confidence", {"number": float(ai_data.get("confidence", 0.0))})
    add_prop_if_exists(props, "Needs Review", {"checkbox": bool(ai_data.get("needs_review", False))})
    add_prop_if_exists(props, "Source Platform", {"select": {"name": ai_data.get("content_type", "Unknown")}})

    if url:
        props["URL"] = {"url": url}
    if ai_data.get("deadline_date"):
        props["Deadline"] = {"date": {"start": ai_data["deadline_date"]}}

    page = notion.pages.create(parent={"database_id": NOTION_DB_ID}, properties=props)
    return page.get("url", "")


# ═════════════════════════════════════════════════════════════
#  STEP 5 — BUILD DISCORD EMBED (reply card)
# ═════════════════════════════════════════════════════════════

CATEGORY_COLORS = {
    "Scholarship": 0x378ADD, "Research": 0x1D9E75, "AI Tools": 0x7F77DD,
    "Tech News": 0xEF9F27,   "Video": 0xD4537E,    "Learning": 0x639922,
    "Social Post": 0x185FA5, "Paper": 0x0F6E56,    "Career": 0x3B6D11,
    "Image": 0x8B7355,       "News": 0xCC5500,      "Template": 0x534AB7,
    "Idea": 0xD4AF37,       "Academic": 0x3B82F6,  "Course": 0x14B8A6,
    "Website": 0x64748B,    "URL List": 0x64748B,  "Reading Note": 0x22C55E,
    "Schedule": 0xF97316,   "Task": 0xF59E0B,      "Fun Chat": 0xEC4899,
    "Archive": 0x6B7280,    "Unknown": 0x991B1B
}

SOURCE_LABELS = {
    "youtube": "📺 YouTube", "facebook": "👤 Facebook",
    "instagram": "📷 Instagram", "tiktok": "🎵 TikTok",
    "twitter": "🐦 X / Twitter", "url": "🔗 Web Article",
    "text": "📝 Text Note", "image": "🖼 Image / Photocard"
}

def build_embed(ai_data: dict, url: str, source_type: str, warning: str = None) -> discord.Embed:
    color = CATEGORY_COLORS.get(ai_data.get("category", ""), 0x5865F2)
    source_label = SOURCE_LABELS.get(source_type, "🔗 Content")

    embed = discord.Embed(
        title=ai_data.get("title", "Saved Content")[:256],
        url=url if url and url.startswith("http") else None,
        color=color
    )

    # Warning for blocked Facebook/Instagram content
    if warning == "blocked":
        embed.add_field(
            name="⚠️ Facebook blocked full content",
            value="Facebook hides posts from bots. **For full text:** copy-paste the post text and use `/note [text]` instead. Link is saved.",
            inline=False
        )
    elif warning == "partial":
        embed.add_field(
            name="ℹ️ Partial content only",
            value="Only the post preview was readable. Use `/note` to add the full text.",
            inline=False
        )

    embed.add_field(name="📝 Summary (English)", value=ai_data.get("summary_en", "—")[:800], inline=False)
    embed.add_field(name="🇧🇩 সারসংক্ষেপ (বাংলা)", value=ai_data.get("summary_bn", "—")[:800], inline=False)

    embed.add_field(name="Category", value=f"`{ai_data.get('category','?')}`", inline=True)
    embed.add_field(name="Type", value=f"`{ai_data.get('content_type','?')}`", inline=True)
    embed.add_field(name="Source", value=source_label, inline=True)
    embed.add_field(name="Relevance", value=f"`{ai_data.get('relevance_score', 5)}/10`", inline=True)
    embed.add_field(name="Confidence", value=f"`{ai_data.get('confidence', 0)}`", inline=True)

    tags_str = "  ".join([f"`{t}`" for t in ai_data.get("sub_tags", [])[:5]])
    if tags_str:
        embed.add_field(name="Tags", value=tags_str, inline=False)

    if ai_data.get("deadline_date"):
        embed.add_field(name="📅 Deadline", value=f"`{ai_data['deadline_date']}`", inline=True)

    embed.add_field(name="Action", value=f"`{ai_data.get('action', 'Save only')}`", inline=True)
    if ai_data.get("needs_review"):
        choices = ", ".join(ai_data.get("possible_categories", [])[:4]) or ", ".join(CATEGORIES[:5])
        embed.add_field(
            name="⚠️ Needs review",
            value=f"I am not fully sure about the class. Possible: `{choices}`\nUse `/fixcategory latest category`.",
            inline=False
        )
    embed.set_footer(text=f"Saved to Notion · AI Campus Assistant · {ai_data.get('language','')}")
    return embed


# ═════════════════════════════════════════════════════════════
#  MASTER SAVE FUNCTION — handles everything
# ═════════════════════════════════════════════════════════════

async def process_and_save(interaction_or_message, content: str, source_type: str = None):
    """
    Single function that handles any content type end-to-end.
    Detects type → fetches readable text → AI categorizes → saves to Notion → replies in Discord.
    """
    is_interaction = isinstance(interaction_or_message, discord.Interaction)

    async def send_reply(text=None, embed=None):
        if is_interaction:
            await interaction_or_message.followup.send(content=text, embed=embed)
        else:
            await interaction_or_message.reply(content=text, embed=embed)

    url = content if content.startswith("http") else ""
    warning = None

    try:
        # ── Detect type if not already known ──────────────────
        if source_type is None:
            source_type = detect_content_type(content)

        # ── Fetch readable text based on source type ──────────
        fetched = {}

        if source_type == "youtube":
            fetched = fetch_youtube(url)
            readable_text = fetched.get("text", url)

        elif source_type == "facebook":
            fetched = fetch_facebook(url)
            readable_text = fetched.get("text", url)
            warning = fetched.get("warning")

        elif source_type == "instagram":
            fetched = fetch_instagram(url)
            readable_text = fetched.get("text", url)

        elif source_type == "tiktok":
            fetched = fetch_tiktok(url)
            readable_text = fetched.get("text", url)

        elif source_type == "twitter":
            fetched = fetch_twitter(url)
            readable_text = fetched.get("text", url)

        elif source_type == "url":
            fetched = fetch_generic_url(url)
            readable_text = fetched.get("text", url)

        elif source_type == "image":
            # content here is the image URL from Discord attachment
            readable_text = extract_text_from_image_gemini(content)
            fetched = {"title": "Image / Photocard", "text": readable_text}

        else:
            # Plain text — use as-is
            readable_text = content
            fetched = {"title": "Text Note", "text": content}

        # ── AI categorization ─────────────────────────────────
        ai_data = categorize_with_ai(readable_text, url=url, source_type=source_type)

        # Override title with YouTube/Facebook title if available
        if fetched.get("title") and fetched["title"] not in ["YouTube Video", "Facebook Post"]:
            if len(fetched["title"]) > 5:  # only if we got a real title
                ai_data["title"] = fetched["title"][:100]

        # ── Save to Notion ────────────────────────────────────
        save_to_notion(ai_data, url, readable_text[:2000])

        # ── Build and send Discord embed ──────────────────────
        embed = build_embed(ai_data, url, source_type, warning)
        await send_reply(text="✅ Saved and organized", embed=embed)

    except Exception as e:
        await send_reply(text=f"❌ Error saving content: `{str(e)}`\n\nIf this keeps happening, try `/note` and paste the text directly.")


# ═════════════════════════════════════════════════════════════
#  DISCORD COMMANDS
# ═════════════════════════════════════════════════════════════

@tree.command(name="save", description="Save any link — YouTube, Facebook, article, research paper")
@app_commands.describe(url="Paste the link you want to save")
async def save_cmd(interaction: discord.Interaction, url: str):
    await interaction.response.defer(thinking=True)
    await process_and_save(interaction, url.strip())


@tree.command(name="note", description="Save copied text — Facebook post text, article text, any notes")
@app_commands.describe(text="Paste the text you copied from Facebook, LinkedIn, anywhere")
async def note_cmd(interaction: discord.Interaction, text: str):
    await interaction.response.defer(thinking=True)
    await process_and_save(interaction, text.strip(), source_type="text")


@tree.command(name="reading_note", description="Save a reading note separately from normal text")
@app_commands.describe(text="Your reading note")
async def reading_note_cmd(interaction: discord.Interaction, text: str):
    await interaction.response.defer(thinking=True)
    ai_data = categorize_with_ai(text.strip(), source_type="note")
    ai_data["category"] = "Reading Note"
    ai_data["content_type"] = "Reading Note"
    ai_data["confidence"] = max(float(ai_data.get("confidence", 0.7)), 0.9)
    ai_data["needs_review"] = False
    save_to_notion(ai_data, "", text.strip())
    await interaction.followup.send(embed=build_embed(ai_data, "", "text"))


@tree.command(name="schedule", description="Save a schedule, reminder, routine, or planned task")
@app_commands.describe(text="Example: tomorrow 8pm read paper")
async def schedule_cmd(interaction: discord.Interaction, text: str):
    await interaction.response.defer(thinking=True)
    ai_data = categorize_with_ai(text.strip(), source_type="schedule")
    ai_data["category"] = "Schedule"
    ai_data["content_type"] = "Schedule"
    ai_data["action"] = "Add to schedule"
    ai_data["confidence"] = max(float(ai_data.get("confidence", 0.7)), 0.9)
    ai_data["needs_review"] = False
    save_to_notion(ai_data, "", text.strip())
    await interaction.followup.send(embed=build_embed(ai_data, "", "text"))


@tree.command(name="ask", description="Ask the assistant without saving to Notion")
@app_commands.describe(question="Your question")
async def ask_cmd(interaction: discord.Interaction, question: str):
    await interaction.response.defer(thinking=True)
    try:
        answer = answer_with_ai(question.strip(), playful=False)
        await interaction.followup.send(answer[:1900])
    except Exception as e:
        await interaction.followup.send(f"❌ Answer error: `{str(e)}`")


@tree.command(name="chat", description="Casual/fun chat without saving to Notion")
@app_commands.describe(message="Say anything")
async def chat_cmd(interaction: discord.Interaction, message: str):
    await interaction.response.defer(thinking=True)
    try:
        answer = answer_with_ai(message.strip(), playful=True)
        await interaction.followup.send(answer[:1900])
    except Exception as e:
        await interaction.followup.send(f"❌ Chat error: `{str(e)}`")


@tree.command(name="fixcategory", description="Fix the category of your latest saved item")
@app_commands.describe(target="Use latest for now", category="New category name, e.g. Scholarship")
async def fixcategory_cmd(interaction: discord.Interaction, target: str, category: str):
    await interaction.response.defer(thinking=True)
    category = normalize_category(category)
    if category not in CATEGORIES:
        await interaction.followup.send(f"Unknown category. Use one of:\n`{', '.join(CATEGORIES)}`")
        return
    if target.lower() != "latest":
        await interaction.followup.send("For now I can fix only `latest`. Example: `/fixcategory latest Scholarship`")
        return
    try:
        results = notion.databases.query(
            database_id=NOTION_DB_ID,
            sorts=[{"property": "Saved At", "direction": "descending"}],
            page_size=1
        )["results"]
        if not results:
            await interaction.followup.send("No saved item found.")
            return
        props = {"Category": {"select": {"name": category}}}
        add_prop_if_exists(props, "Needs Review", {"checkbox": False})
        add_prop_if_exists(props, "Confidence", {"number": 1.0})
        notion.pages.update(page_id=results[0]["id"], properties=props)
        await interaction.followup.send(f"✅ Latest item category updated to `{category}`.")
    except Exception as e:
        await interaction.followup.send(f"❌ Category update error: `{str(e)}`")


@tree.command(name="find", description="Search your saved content with natural language")
@app_commands.describe(query="What are you looking for? (e.g. scholarship UK 2025, free AI tools)")
async def find_cmd(interaction: discord.Interaction, query: str):
    await interaction.response.defer(thinking=True)
    try:
        results = notion.databases.query(
            database_id=NOTION_DB_ID,
            filter={"property": "Title", "title": {"contains": query}},
            sorts=[{"property": "Saved At", "direction": "descending"}],
            page_size=5
        )["results"]

        if not results:
            await interaction.followup.send(
                f"Nothing found for **'{query}'**.\n\nTry a shorter keyword. Example: instead of 'fully funded UK scholarship' try just 'scholarship' or 'UK'."
            )
            return

        embeds = []
        for r in results[:4]:
            p = r["properties"]
            title    = p["Title"]["title"][0]["plain_text"] if p["Title"]["title"] else "Untitled"
            cat      = p["Category"]["select"]["name"] if p.get("Category") and p["Category"].get("select") else "?"
            summary  = p["Summary EN"]["rich_text"][0]["plain_text"] if p.get("Summary EN") and p["Summary EN"]["rich_text"] else ""
            url_val  = p.get("URL", {}).get("url") or ""
            lang     = p["Language"]["select"]["name"] if p.get("Language") and p["Language"].get("select") else ""
            color    = CATEGORY_COLORS.get(cat, 0x5865F2)

            e = discord.Embed(title=title, url=url_val or None, color=color)
            if summary:
                e.add_field(name="Summary", value=summary[:400], inline=False)
            e.add_field(name="Category", value=f"`{cat}`", inline=True)
            if lang:
                e.add_field(name="Language", value=f"`{lang}`", inline=True)
            embeds.append(e)

        await interaction.followup.send(
            f"Found **{len(results)}** result(s) for **'{query}'**:",
            embeds=embeds
        )
    except Exception as e:
        await interaction.followup.send(f"❌ Search error: `{str(e)}`")


@tree.command(name="list", description="Show all saved content in one category")
@app_commands.describe(category="Category name — Scholarship, Research, AI Tools, Video, Social Post, etc.")
async def list_cmd(interaction: discord.Interaction, category: str):
    await interaction.response.defer(thinking=True)
    try:
        category = normalize_category(category)
        results = notion.databases.query(
            database_id=NOTION_DB_ID,
            filter={"property": "Category", "select": {"equals": category}},
            sorts=[{"property": "Saved At", "direction": "descending"}],
            page_size=10
        )["results"]

        if not results:
            all_cats = "Scholarship, Research, AI Tools, Tech News, Learning, Career, Video, Image, Social Post, Paper, Template, Idea, News"
            await interaction.followup.send(
                f"Nothing saved in **'{category}'** yet.\n\nAvailable categories:\n`{all_cats}`"
            )
            return

        lines = []
        for r in results[:10]:
            p = r["properties"]
            title   = p["Title"]["title"][0]["plain_text"] if p["Title"]["title"] else "Untitled"
            url_val = p.get("URL", {}).get("url") or ""
            lines.append(f"• [{title}]({url_val})" if url_val else f"• {title}")

        e = discord.Embed(
            title=f"{category} — {len(results)} saved",
            color=CATEGORY_COLORS.get(category, 0x5865F2)
        )
        e.description = "\n".join(lines)
        await interaction.followup.send(embed=e)
    except Exception as ex:
        await interaction.followup.send(f"❌ Error: `{str(ex)}`")


@tree.command(name="deadlines", description="Show all upcoming deadlines from saved scholarships and applications")
async def deadlines_cmd(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    try:
        results = notion.databases.query(
            database_id=NOTION_DB_ID,
            filter={"property": "Has Deadline", "checkbox": {"equals": True}},
            sorts=[{"property": "Deadline", "direction": "ascending"}],
            page_size=10
        )["results"]

        if not results:
            await interaction.followup.send(
                "No items with deadlines saved yet.\n\nWhen you save a scholarship or application, the bot automatically detects the deadline date."
            )
            return

        e = discord.Embed(title="📅 Upcoming Deadlines", color=0xE24B4A)
        for r in results:
            p = r["properties"]
            title = p["Title"]["title"][0]["plain_text"] if p["Title"]["title"] else "Untitled"
            dl    = p["Deadline"]["date"]["start"] if p.get("Deadline") and p["Deadline"].get("date") else "Unknown"
            e.add_field(name=title, value=f"Deadline: `{dl}`", inline=False)
        await interaction.followup.send(embed=e)
    except Exception as ex:
        await interaction.followup.send(f"❌ Error: `{str(ex)}`")


@tree.command(name="recent", description="Show recently saved items")
@app_commands.describe(count="How many items to show (max 10, default 5)")
async def recent_cmd(interaction: discord.Interaction, count: int = 5):
    await interaction.response.defer(thinking=True)
    try:
        results = notion.databases.query(
            database_id=NOTION_DB_ID,
            sorts=[{"property": "Saved At", "direction": "descending"}],
            page_size=min(count, 10)
        )["results"]

        lines = []
        for r in results:
            p     = r["properties"]
            title = p["Title"]["title"][0]["plain_text"] if p["Title"]["title"] else "Untitled"
            cat   = p["Category"]["select"]["name"] if p.get("Category") and p["Category"].get("select") else "?"
            lines.append(f"• `{cat}` — {title}")

        e = discord.Embed(title=f"Last {len(results)} saved items", color=0x378ADD)
        e.description = "\n".join(lines)
        await interaction.followup.send(embed=e)
    except Exception as ex:
        await interaction.followup.send(f"❌ Error: `{str(ex)}`")


@tree.command(name="stats", description="See your content library statistics")
async def stats_cmd(interaction: discord.Interaction):
    await interaction.response.defer(thinking=True)
    try:
        results = notion.databases.query(
            database_id=NOTION_DB_ID,
            page_size=100
        )["results"]

        counts = {}
        for r in results:
            p   = r["properties"]
            cat = p["Category"]["select"]["name"] if p.get("Category") and p["Category"].get("select") else "Other"
            counts[cat] = counts.get(cat, 0) + 1

        e = discord.Embed(title=f"📊 Your Content Library — {len(results)} items total", color=0x1D9E75)
        for cat, n in sorted(counts.items(), key=lambda x: -x[1]):
            bar = "█" * min(n, 20)
            e.add_field(name=f"{cat} ({n})", value=bar or "▏", inline=False)
        await interaction.followup.send(embed=e)
    except Exception as ex:
        await interaction.followup.send(f"❌ Error: `{str(ex)}`")


@tree.command(name="help", description="Show all commands and how to use them")
async def help_cmd(interaction: discord.Interaction):
    e = discord.Embed(title="🤖 Campus Assistant — Commands", color=0x5865F2)
    e.add_field(name="/save [url]", value="Save any link — YouTube, articles, research papers\nFacebook/Instagram links: saves what it can", inline=False)
    e.add_field(name="/note [text]", value="**Best for Facebook posts** — copy the post text and paste here\nAlso for LinkedIn, TikTok captions, any copied text", inline=False)
    e.add_field(name="/reading_note [text]", value="Save study notes separately as Reading Note", inline=False)
    e.add_field(name="/schedule [text]", value="Save routines, deadlines, reminders, or tasks", inline=False)
    e.add_field(name="/ask [question]", value="Ask something without saving it", inline=False)
    e.add_field(name="/chat [message]", value="Casual/fun chat without saving it", inline=False)
    e.add_field(name="/fixcategory latest [category]", value="Fix the latest saved item's class if AI was unsure", inline=False)
    e.add_field(name="📸 Drop an image", value="Upload any image or photocard directly in this channel\nBot reads all text (Bangla + English) using AI vision", inline=False)
    e.add_field(name="/find [query]", value="Search your saved content: `/find scholarship UK`", inline=False)
    e.add_field(name="/list [category]", value="List by category: `/list Scholarship` `/list Video`", inline=False)
    e.add_field(name="/deadlines", value="Show all items with upcoming deadlines", inline=False)
    e.add_field(name="/recent [n]", value="Show last N saved items", inline=False)
    e.add_field(name="/stats", value="See your library breakdown by category", inline=False)
    e.add_field(
        name="⚠️ Facebook tip",
        value="Facebook blocks link reading. For best results:\n1. Copy the post text on Facebook\n2. Use `/note [paste text here]`",
        inline=False
    )
    await interaction.response.send_message(embed=e)


# ═════════════════════════════════════════════════════════════
#  AUTO-PROCESS IMAGES DROPPED IN CHANNEL
# ═════════════════════════════════════════════════════════════

@bot.event
async def on_message(message: discord.Message):
    """
    Automatically process images when user drops them in the channel.
    No command needed — just upload the image.
    """
    if message.author.bot:
        return

    for attachment in message.attachments:
        filename_lower = attachment.filename.lower()
        is_image = any(filename_lower.endswith(ext) for ext in [
            ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tiff"
        ])

        if is_image:
            # Show thinking reaction while processing
            try:
                await message.add_reaction("⏳")
            except Exception:
                pass

            await process_and_save(message, attachment.url, source_type="image")

            try:
                await message.remove_reaction("⏳", bot.user)
                await message.add_reaction("✅")
            except Exception:
                pass


# ═════════════════════════════════════════════════════════════
#  STARTUP
# ═════════════════════════════════════════════════════════════

@bot.event
async def on_ready():
    await tree.sync()
    print(f"✅ Campus Assistant online: {bot.user}")
    print(f"   Serving {len(bot.guilds)} server(s)")


bot.run(DISCORD_TOKEN)
