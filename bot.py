import discord
from discord import app_commands
import requests, json, os
from notion_client import Client
from datetime import datetime

DISCORD_TOKEN = os.environ["DISCORD_TOKEN"]
GROQ_KEY      = os.environ["GROQ_API_KEY"]
NOTION_SECRET = os.environ["NOTION_SECRET"]
NOTION_DB_ID  = os.environ["NOTION_DB_ID"]

notion = Client(auth=NOTION_SECRET)
intents = discord.Intents.default()
intents.message_content = True
bot = discord.Client(intents=intents)
tree = app_commands.CommandTree(bot)

# ── AI categorization ─────────────────────────────────────────
def categorize(text, url=""):
    prompt = f"""You are a personal content organizer for a Bangladeshi AI student named Marof.
Content: {text[:1500]}
URL: {url}
Return ONLY valid JSON, no other text:
{{"title":"short title max 10 words","category":"Scholarship|Research|AI Tools|Tech News|Learning|Career|Video|Image|Social Post|Paper|Template|Idea","sub_tags":["t1","t2","t3"],"language":"Bangla|English|Both","summary_en":"2-sentence English summary","summary_bn":"2-sentence Bangla summary in Bengali script","relevance_score":8.0,"has_deadline":false,"deadline_date":null,"action":"Apply|Read|Watch|Save only","keywords":["k1","k2","k3"]}}"""
    
    r = requests.post("https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization":f"Bearer {GROQ_KEY}","Content-Type":"application/json"},
        json={"model":"llama-3.1-8b-instant","messages":[{"role":"user","content":prompt}],"max_tokens":600},
        timeout=20)
    return json.loads(r.json()["choices"][0]["message"]["content"])

# ── Save to Notion ────────────────────────────────────────────
def save_notion(d, url, raw):
    tags = [{"name":t} for t in d.get("sub_tags",[])]
    props = {
        "Title":      {"title":[{"text":{"content":d["title"]}}]},
        "Category":   {"select":{"name":d["category"]}},
        "Tags":       {"multi_select":tags},
        "Language":   {"select":{"name":d["language"]}},
        "Summary EN": {"rich_text":[{"text":{"content":d["summary_en"]}}]},
        "Summary BN": {"rich_text":[{"text":{"content":d["summary_bn"]}}]},
        "Relevance":  {"number":float(d.get("relevance_score",5))},
        "Action":     {"select":{"name":d.get("action","Save only")}},
        "Has Deadline":{"checkbox":d.get("has_deadline",False)},
        "Raw Text":   {"rich_text":[{"text":{"content":raw[:2000]}}]},
        "Saved At":   {"date":{"start":datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")}},
    }
    if url: props["URL"] = {"url":url}
    if d.get("deadline_date"): props["Deadline"] = {"date":{"start":d["deadline_date"]}}
    notion.pages.create(parent={"database_id":NOTION_DB_ID}, properties=props)

# ── Build embed ───────────────────────────────────────────────
def make_embed(d, url=""):
    COLORS = {"Scholarship":0x378ADD,"Research":0x1D9E75,"AI Tools":0x7F77DD,
              "Tech News":0xEF9F27,"Video":0xD4537E,"Career":0x639922,"Paper":0x0F6E56}
    e = discord.Embed(title=d["title"], url=url or None, color=COLORS.get(d.get("category",""),0x5865F2))
    e.add_field(name="Summary", value=d["summary_en"], inline=False)
    e.add_field(name="বাংলা সারসংক্ষেপ", value=d["summary_bn"], inline=False)
    e.add_field(name="Category", value=f'`{d["category"]}`', inline=True)
    e.add_field(name="Language", value=f'`{d["language"]}`', inline=True)
    e.add_field(name="Relevance", value=f'`{d.get("relevance_score",5)}/10`', inline=True)
    tags = " ".join([f"`{t}`" for t in d.get("sub_tags",[])])
    if tags: e.add_field(name="Tags", value=tags, inline=False)
    if d.get("deadline_date"): e.add_field(name="Deadline", value=f'`{d["deadline_date"]}`', inline=True)
    e.set_footer(text="Saved to Notion · AI Campus Assistant")
    return e

# ── /save ─────────────────────────────────────────────────────
@tree.command(name="save", description="Save a URL to your content vault")
@app_commands.describe(url="URL to save")
async def save_cmd(interaction, url: str):
    await interaction.response.defer()
    try:
        d = categorize(f"URL: {url}", url)
        save_notion(d, url, url)
        await interaction.followup.send("Saved and organized ✓", embed=make_embed(d, url))
    except Exception as ex:
        await interaction.followup.send(f"Error: {ex}")

# ── /note ─────────────────────────────────────────────────────
@tree.command(name="note", description="Save copied text or a post")
@app_commands.describe(text="Text to save")
async def note_cmd(interaction, text: str):
    await interaction.response.defer()
    try:
        d = categorize(text)
        save_notion(d, "", text)
        await interaction.followup.send("Saved and organized ✓", embed=make_embed(d))
    except Exception as ex:
        await interaction.followup.send(f"Error: {ex}")

# ── /find ─────────────────────────────────────────────────────
@tree.command(name="find", description="Search your saved content")
@app_commands.describe(query="What to search for")
async def find_cmd(interaction, query: str):
    await interaction.response.defer()
    try:
        results = notion.databases.query(
            database_id=NOTION_DB_ID,
            filter={"property":"Title","title":{"contains":query}},
            sorts=[{"property":"Saved At","direction":"descending"}],
            page_size=4)["results"]
        if not results:
            await interaction.followup.send(f"Nothing found for '{query}'"); return
        embeds = []
        for r in results:
            p = r["properties"]
            title = p["Title"]["title"][0]["plain_text"] if p["Title"]["title"] else "Untitled"
            cat   = p["Category"]["select"]["name"] if p["Category"]["select"] else "?"
            s_en  = p["Summary EN"]["rich_text"][0]["plain_text"] if p["Summary EN"]["rich_text"] else ""
            url   = p.get("URL",{}).get("url") or ""
            e = discord.Embed(title=title, url=url or None, color=0x1D9E75)
            e.add_field(name="Category", value=f"`{cat}`", inline=True)
            if s_en: e.add_field(name="Summary", value=s_en[:200], inline=False)
            embeds.append(e)
        await interaction.followup.send(f"Found {len(results)} results for '{query}':", embeds=embeds)
    except Exception as ex:
        await interaction.followup.send(f"Error: {ex}")

# ── /list ─────────────────────────────────────────────────────
@tree.command(name="list", description="List saved content by category")
@app_commands.describe(category="Category (Scholarship, Research, AI Tools, etc.)")
async def list_cmd(interaction, category: str):
    await interaction.response.defer()
    try:
        results = notion.databases.query(
            database_id=NOTION_DB_ID,
            filter={"property":"Category","select":{"equals":category.title()}},
            sorts=[{"property":"Saved At","direction":"descending"}],
            page_size=8)["results"]
        if not results:
            await interaction.followup.send(f"Nothing in '{category}' yet."); return
        lines = []
        for r in results:
            p = r["properties"]
            t = p["Title"]["title"][0]["plain_text"] if p["Title"]["title"] else "Untitled"
            u = p.get("URL",{}).get("url") or ""
            lines.append(f"• [{t}]({u})" if u else f"• {t}")
        e = discord.Embed(title=f"{category.title()} — {len(results)} saved", color=0x5865F2)
        e.description = "\n".join(lines)
        await interaction.followup.send(embed=e)
    except Exception as ex:
        await interaction.followup.send(f"Error: {ex}")

# ── /deadlines ────────────────────────────────────────────────
@tree.command(name="deadlines", description="Show upcoming deadlines")
async def deadlines_cmd(interaction):
    await interaction.response.defer()
    try:
        results = notion.databases.query(
            database_id=NOTION_DB_ID,
            filter={"property":"Has Deadline","checkbox":{"equals":True}},
            sorts=[{"property":"Deadline","direction":"ascending"}],
            page_size=8)["results"]
        if not results:
            await interaction.followup.send("No deadline items saved yet."); return
        e = discord.Embed(title="Upcoming Deadlines", color=0xE24B4A)
        for r in results:
            p = r["properties"]
            title = p["Title"]["title"][0]["plain_text"] if p["Title"]["title"] else "Untitled"
            dl = p["Deadline"]["date"]["start"] if p["Deadline"]["date"] else "Unknown"
            e.add_field(name=title, value=f"`{dl}`", inline=False)
        await interaction.followup.send(embed=e)
    except Exception as ex:
        await interaction.followup.send(f"Error: {ex}")

# ── /recent ───────────────────────────────────────────────────
@tree.command(name="recent", description="Show recently saved items")
@app_commands.describe(count="How many to show (default 5)")
async def recent_cmd(interaction, count: int = 5):
    await interaction.response.defer()
    try:
        results = notion.databases.query(
            database_id=NOTION_DB_ID,
            sorts=[{"property":"Saved At","direction":"descending"}],
            page_size=min(count,8))["results"]
        lines = []
        for r in results:
            p = r["properties"]
            title = p["Title"]["title"][0]["plain_text"] if p["Title"]["title"] else "Untitled"
            cat   = p["Category"]["select"]["name"] if p["Category"]["select"] else "?"
            lines.append(f"• `{cat}` — {title}")
        e = discord.Embed(title=f"Last {len(results)} saved items", color=0x378ADD)
        e.description = "\n".join(lines)
        await interaction.followup.send(embed=e)
    except Exception as ex:
        await interaction.followup.send(f"Error: {ex}")

# ── Auto-process image uploads ────────────────────────────────
@bot.event
async def on_message(message):
    if message.author.bot: return
    for att in message.attachments:
        if any(att.filename.lower().endswith(x) for x in ['.png','.jpg','.jpeg','.webp','.gif']):
            await message.add_reaction("⏳")
            try:
                d = categorize(f"Image: {att.filename} uploaded to Discord", att.url)
                save_notion(d, att.url, f"Image: {att.filename}")
                await message.reply("Image saved and categorized ✓", embed=make_embed(d, att.url))
            except Exception as ex:
                await message.reply(f"Could not process image: {ex}")
            await message.remove_reaction("⏳", bot.user)

# ── Startup ───────────────────────────────────────────────────
@bot.event
async def on_ready():
    await tree.sync()
    print(f"Campus Assistant online: {bot.user}")

bot.run(DISCORD_TOKEN)
