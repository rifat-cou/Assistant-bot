BOT_NAME = "Campus Assistant"

SHORT_DESCRIPTION = (
    "Personal AI vault for links, screenshots, videos, notes, schedules, scholarships, "
    "research, courses, and AI tools."
)

LONG_DESCRIPTION = """Campus Assistant is Marof's personal AI organizer.

Send links, screenshots, videos, PDFs, voice notes, copied Facebook posts, reading notes, schedules, courses, scholarship info, research topics, AI tools, and useful websites.

The bot reads the content, separates content type from topic, summarizes in English and Bangla, extracts deadlines/tools/URLs, asks for review when unsure, and saves everything to Notion.

Use /ask or /chat when you want to talk without saving anything."""

SERVER_DESCRIPTION = """A private AI knowledge hub for collecting and organizing useful content.

Use this server to save:
- Facebook/YouTube/social posts
- Screenshots and website captures
- Research and academic materials
- Scholarship and career opportunities
- AI tools, courses, templates, and ideas
- Reading notes, schedules, and tasks

Everything important goes into Notion with summaries, tags, deadlines, source type, category, and review status."""

DISCORD_COMMANDS = [
    ("/save <url>", "Save any link: YouTube, Facebook, article, website, paper, course, or tool."),
    ("/note <text>", "Save copied text, Facebook post text, captions, or quick notes."),
    ("/reading_note <text>", "Save study notes, book notes, paper notes, or learning reflections."),
    ("/schedule <text>", "Save a routine, deadline, reminder, plan, or task."),
    ("/ask <question>", "Ask a question without saving it to Notion."),
    ("/chat <message>", "Casual or fun conversation without saving it."),
    ("/find <query>", "Search your saved library by keyword or natural language."),
    ("/list <category>", "Show saved items from one category."),
    ("/deadlines", "Show saved items with upcoming deadlines."),
    ("/recent <count>", "Show recently saved items."),
    ("/stats", "Show your library category breakdown."),
    ("/fixcategory latest <category>", "Correct the latest saved item's category."),
    ("/about", "Show what the assistant can do."),
    ("/help", "Show all commands."),
]

TELEGRAM_COMMANDS = [
    ("start", "Open the assistant intro"),
    ("help", "Show all commands"),
    ("about", "What this assistant does"),
    ("ask", "Ask without saving"),
    ("chat", "Casual chat without saving"),
    ("note", "Save a reading note"),
    ("schedule", "Save schedule or task"),
    ("find", "Search saved content"),
    ("list", "List one category"),
    ("deadlines", "Show upcoming deadlines"),
    ("stats", "Show library stats"),
    ("fixcategory", "Fix latest item category"),
]


def command_help_text(platform: str = "telegram") -> str:
    lines = [
        f"{BOT_NAME} Commands",
        "",
        "Save by sending links, screenshots, videos, PDFs, voice notes, or copied text.",
        "",
    ]
    commands = DISCORD_COMMANDS if platform == "discord" else [
        (f"/{name}", desc) for name, desc in TELEGRAM_COMMANDS
    ]
    for command, description in commands:
        lines.append(f"{command} - {description}")
    lines.extend(
        [
            "",
            "Tip: Facebook often blocks direct reading. Send a screenshot or copy-paste the post text for best results.",
        ]
    )
    return "\n".join(lines)


def about_text() -> str:
    return f"{BOT_NAME}\n\n{LONG_DESCRIPTION}"
