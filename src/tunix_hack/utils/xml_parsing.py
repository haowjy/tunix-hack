"""XML tag extraction utilities for <reasoning>/<answer> format."""


def extract_tag(text: str, tag: str) -> str:
    """Extract content between XML-style tags.

    Args:
        text: The text containing the tags.
        tag: The tag name (e.g., "reasoning" or "answer").

    Returns:
        The content between the tags, or empty string if not found.
    """
    start = text.find(f"<{tag}>")
    end = text.find(f"</{tag}>")
    if start == -1 or end == -1:
        return ""
    return text[start + len(tag) + 2 : end].strip()


def has_valid_format(text: str) -> bool:
    """Check if text has valid <reasoning>/<answer> format.

    Args:
        text: The text to check.

    Returns:
        True if both tags are present.
    """
    reasoning = extract_tag(text, "reasoning")
    answer = extract_tag(text, "answer")
    return bool(reasoning) and bool(answer)
