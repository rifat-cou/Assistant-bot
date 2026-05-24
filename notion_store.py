from datetime import datetime
import os

from notion_client import Client

from classifier import CATEGORIES, normalize_category


class NotionStore:
    """Shared Notion database access for saving, searching, and fixing items."""

    def __init__(self, notion_secret=None, database_id=None):
        self.database_id = database_id or os.getenv("NOTION_DB_ID")
        self.client = Client(auth=notion_secret or os.getenv("NOTION_SECRET"))
        self._property_cache = None

    def properties(self) -> set:
        if self._property_cache is None:
            try:
                db = self.client.databases.retrieve(database_id=self.database_id)
                self._property_cache = set(db.get("properties", {}).keys())
            except Exception:
                self._property_cache = set()
        return self._property_cache

    def add_if_exists(self, props: dict, name: str, value: dict):
        if name in self.properties():
            props[name] = value

    def save(self, ai_data: dict, source_url: str = "", raw_text: str = "", file_type: str = "") -> str:
        tags = [{"name": t} for t in ai_data.get("sub_tags", [])[:10]]
        for url in ai_data.get("extracted_urls", [])[:3]:
            if url and len(url) < 50:
                tags.append({"name": f"url:{url[:40]}"})

        props = {
            "Title": {"title": [{"text": {"content": ai_data.get("title", "Untitled")[:100]}}]},
            "Category": {"select": {"name": ai_data.get("category", "Unknown")}},
            "Tags": {"multi_select": tags},
            "Language": {"select": {"name": ai_data.get("language", "Both")}},
            "Summary EN": {"rich_text": [{"text": {"content": ai_data.get("summary_en", "")[:2000]}}]},
            "Summary BN": {"rich_text": [{"text": {"content": ai_data.get("summary_bn", "")[:2000]}}]},
            "Relevance": {"number": float(ai_data.get("relevance_score", 5.0))},
            "Action": {"select": {"name": ai_data.get("action", "Save only")}},
            "Has Deadline": {"checkbox": bool(ai_data.get("has_deadline", False))},
            "Raw Text": {"rich_text": [{"text": {"content": raw_text[:2000]}}]},
            "Saved At": {"date": {"start": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")}},
        }
        self.add_if_exists(props, "Content Type", {"select": {"name": ai_data.get("content_type", file_type or "Unknown")}})
        self.add_if_exists(props, "Confidence", {"number": float(ai_data.get("confidence", 0.0))})
        self.add_if_exists(props, "Needs Review", {"checkbox": bool(ai_data.get("needs_review", False))})
        self.add_if_exists(props, "Source Platform", {"select": {"name": file_type or ai_data.get("content_type", "Unknown")}})

        if source_url:
            props["URL"] = {"url": source_url}
        if ai_data.get("deadline_date"):
            props["Deadline"] = {"date": {"start": ai_data["deadline_date"]}}

        page = self.client.pages.create(parent={"database_id": self.database_id}, properties=props)
        return page.get("url", "")

    def search_title(self, query: str, limit: int = 5) -> list:
        return self.client.databases.query(
            database_id=self.database_id,
            filter={"property": "Title", "title": {"contains": query}},
            sorts=[{"property": "Saved At", "direction": "descending"}],
            page_size=limit,
        )["results"]

    def list_category(self, category: str, limit: int = 10) -> list:
        category = normalize_category(category)
        return self.client.databases.query(
            database_id=self.database_id,
            filter={"property": "Category", "select": {"equals": category}},
            sorts=[{"property": "Saved At", "direction": "descending"}],
            page_size=limit,
        )["results"]

    def recent(self, limit: int = 10) -> list:
        return self.client.databases.query(
            database_id=self.database_id,
            sorts=[{"property": "Saved At", "direction": "descending"}],
            page_size=limit,
        )["results"]

    def upcoming_deadlines(self, limit: int = 10) -> list:
        return self.client.databases.query(
            database_id=self.database_id,
            filter={"property": "Has Deadline", "checkbox": {"equals": True}},
            sorts=[{"property": "Deadline", "direction": "ascending"}],
            page_size=limit,
        )["results"]

    def stats(self) -> dict:
        results = self.client.databases.query(database_id=self.database_id, page_size=100)["results"]
        counts = {}
        for row in results:
            props = row["properties"]
            cat = props["Category"]["select"]["name"] if props.get("Category") and props["Category"].get("select") else "Other"
            counts[cat] = counts.get(cat, 0) + 1
        return {"total": len(results), "counts": counts}

    def fix_latest_category(self, category: str) -> bool:
        category = normalize_category(category)
        if category not in CATEGORIES:
            raise ValueError(f"Unknown category: {category}")
        results = self.recent(limit=1)
        if not results:
            return False
        props = {"Category": {"select": {"name": category}}}
        self.add_if_exists(props, "Needs Review", {"checkbox": False})
        self.add_if_exists(props, "Confidence", {"number": 1.0})
        self.client.pages.update(page_id=results[0]["id"], properties=props)
        return True
