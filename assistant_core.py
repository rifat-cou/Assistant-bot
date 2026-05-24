"""
Compatibility layer.

The project now uses split shared modules. Older imports from assistant_core.py
continue to work so Discord and Telegram can be refactored gradually.
"""

from classifier import (  # noqa: F401
    ACTIONS,
    CATEGORIES,
    CONTENT_TYPES,
    build_categorize_prompt,
    clean_json_response,
    default_possible_categories,
    infer_content_type,
    normalize_ai_data,
    normalize_category,
    split_long_message,
)
