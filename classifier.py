import json
import re


CATEGORIES = [
    "Scholarship",
    "Research",
    "Academic",
    "Course",
    "AI Tools",
    "Tech News",
    "Learning",
    "Career",
    "Video",
    "Image",
    "Social Post",
    "Paper",
    "Template",
    "Idea",
    "News",
    "Website",
    "URL List",
    "Reading Note",
    "Schedule",
    "Task",
    "Fun Chat",
    "Archive",
    "Unknown",
]

CONTENT_TYPES = [
    "Facebook Post",
    "YouTube Video",
    "Instagram Post",
    "TikTok Video",
    "Tweet",
    "Website",
    "Screenshot",
    "Image",
    "Video",
    "PDF",
    "Voice Note",
    "Reading Note",
    "Schedule",
    "Text Note",
    "URL",
    "Unknown",
]

ACTIONS = ["Apply", "Read", "Watch", "Save only", "Visit", "Review", "Add to schedule", "Reply"]


def build_categorize_prompt(text: str, source: str = "", url: str = "") -> str:
    return f"""You are Marof's personal content organizer. Marof is a Bangladeshi student building AI Campus.

Your job is to classify content intelligently for a Notion knowledge vault.

Important:
- Separate CONTENT TYPE from CATEGORY.
- CONTENT TYPE means the form: screenshot, website, video, PDF, Facebook post, reading note, schedule.
- CATEGORY means the topic/purpose: Scholarship, Research, AI Tools, Course, Academic, Reading Note, Schedule, etc.
- A screenshot about scholarship is content_type "Screenshot" and category "Scholarship".
- If unsure, set confidence below 0.70 and needs_review true.
- Detect deadlines, schedules, tasks, links, tool names, and next action.

Source type: {source}
URL: {url}
Content text:
{text[:3000]}

Return ONLY valid JSON, no markdown, no explanation:
{{
  "title": "clear descriptive title in English, max 12 words",
  "category": "pick exactly one: {' | '.join(CATEGORIES)}",
  "content_type": "pick exactly one: {' | '.join(CONTENT_TYPES)}",
  "confidence": 0.85,
  "needs_review": false,
  "possible_categories": ["only include 2-4 options if unsure"],
  "sub_tags": ["tag1", "tag2", "tag3"],
  "language": "Bangla | English | Both",
  "summary_en": "2 clear sentences summarizing the content in English",
  "summary_bn": "২টি বাক্যে বাংলায় সারসংক্ষেপ",
  "relevance_score": 8.0,
  "relevance_reason": "one sentence why this matters to Marof",
  "has_deadline": false,
  "deadline_date": null,
  "action": "pick exactly one: {' | '.join(ACTIONS)}",
  "extracted_urls": ["URLs or website names found"],
  "extracted_tools": ["AI tools, apps, platforms, software names found"],
  "keywords": ["keyword1", "keyword2", "keyword3", "keyword4"]
}}"""


def clean_json_response(raw: str) -> dict:
    raw = raw.replace("```json", "").replace("```", "").strip()
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if match:
        raw = match.group(0)
    return json.loads(raw)


def normalize_category(value: str) -> str:
    cleaned = (value or "").strip().lower().replace("_", " ").replace("-", " ")
    for category in CATEGORIES:
        if category.lower() == cleaned:
            return category
    aliases = {
        "ai tool": "AI Tools",
        "ai tools": "AI Tools",
        "tech": "Tech News",
        "url": "Website",
        "urls": "URL List",
        "note": "Reading Note",
        "reading": "Reading Note",
        "todo": "Task",
        "unknown": "Unknown",
    }
    return aliases.get(cleaned, value if value in CATEGORIES else "Unknown")


def infer_content_type(source: str = "") -> str:
    mapping = {
        "youtube": "YouTube Video",
        "facebook": "Facebook Post",
        "instagram": "Instagram Post",
        "tiktok": "TikTok Video",
        "twitter": "Tweet",
        "url": "Website",
        "web": "Website",
        "image": "Screenshot",
        "screenshot": "Screenshot",
        "video": "Video",
        "pdf": "PDF",
        "voice": "Voice Note",
        "schedule": "Schedule",
        "note": "Reading Note",
        "text": "Text Note",
    }
    return mapping.get((source or "").lower(), "Unknown")


def default_possible_categories(source: str = "", text: str = "") -> list:
    lower = f"{source} {text[:500]}".lower()
    if any(w in lower for w in ["scholarship", "deadline", "apply", "fully funded"]):
        return ["Scholarship", "Academic", "Career"]
    if any(w in lower for w in ["course", "class", "lecture", "lesson"]):
        return ["Course", "Learning", "Academic"]
    if any(w in lower for w in ["tool", "ai", "app", "software"]):
        return ["AI Tools", "Website", "Tech News"]
    if any(w in lower for w in ["schedule", "routine", "tomorrow", "today"]):
        return ["Schedule", "Task", "Academic"]
    return ["Academic", "Learning", "Archive"]


def normalize_ai_data(data: dict, source: str = "", text: str = "") -> dict:
    if not isinstance(data, dict):
        data = {}

    category = normalize_category(data.get("category") or "Unknown")
    content_type = data.get("content_type") or infer_content_type(source)
    if content_type not in CONTENT_TYPES:
        content_type = infer_content_type(source)

    try:
        confidence = float(data.get("confidence", 0.7))
    except Exception:
        confidence = 0.7
    confidence = max(0.0, min(1.0, confidence))

    needs_review = bool(data.get("needs_review", False)) or confidence < 0.7 or category == "Unknown"

    sub_tags = data.get("sub_tags") or []
    if isinstance(sub_tags, str):
        sub_tags = [sub_tags]

    possible = data.get("possible_categories") or []
    if isinstance(possible, str):
        possible = [possible]
    possible = [c for c in possible if c in CATEGORIES][:4]
    if needs_review and not possible:
        possible = default_possible_categories(source, text)

    return {
        **data,
        "title": data.get("title") or f"Saved {content_type}",
        "category": category,
        "content_type": content_type,
        "confidence": confidence,
        "needs_review": needs_review,
        "possible_categories": possible,
        "sub_tags": sub_tags[:10],
        "language": data.get("language") or "Both",
        "summary_en": data.get("summary_en") or text[:240],
        "summary_bn": data.get("summary_bn") or "বিষয়বস্তু সংরক্ষিত হয়েছে।",
        "relevance_score": data.get("relevance_score", 5.0),
        "has_deadline": bool(data.get("has_deadline", False)),
        "deadline_date": data.get("deadline_date"),
        "action": data.get("action") if data.get("action") in ACTIONS else "Save only",
        "extracted_urls": data.get("extracted_urls") or [],
        "extracted_tools": data.get("extracted_tools") or [],
        "keywords": data.get("keywords") or [],
    }


def split_long_message(text: str, limit: int = 3900) -> list:
    if len(text) <= limit:
        return [text]
    chunks = []
    remaining = text
    while remaining:
        chunk = remaining[:limit]
        split_at = max(chunk.rfind("\n"), chunk.rfind(". "), chunk.rfind(" "))
        if split_at < limit * 0.5:
            split_at = limit
        chunks.append(remaining[:split_at].strip())
        remaining = remaining[split_at:].strip()
    return chunks
