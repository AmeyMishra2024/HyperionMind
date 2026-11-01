from __future__ import annotations

import re


NAME_PATTERN = re.compile(r"\b(?:my name is|call me|i am|i'm)\s+([A-Za-z][A-Za-z'-]{1,30})", re.I)


def remembered_name(history: list[dict[str, str]]) -> str | None:
    for message in reversed(history):
        if message.get("role") != "user":
            continue
        match = NAME_PATTERN.search(message.get("content", ""))
        if match:
            return match.group(1).capitalize()
    return None


def demo_respond(history: list[dict[str, str]], user_text: str) -> str:
    """Small deterministic conversation engine for testing without network credentials."""
    text = " ".join(user_text.split()).strip()
    lower = text.lower()
    current_name_match = NAME_PATTERN.search(text)
    name = current_name_match.group(1).capitalize() if current_name_match else remembered_name(history)

    if current_name_match:
        return f"Nice to meet you, {name}. What would you like to talk about?"
    if "what is my name" in lower or "remember my name" in lower or "who am i" in lower:
        return f"You told me your name is {name}." if name else "You have not told me your name yet."
    if any(phrase in lower for phrase in ("how are you", "how's it going", "how is it going")):
        return "I'm doing well and enjoying meeting everyone here. How are you feeling today?"
    if any(word in lower for word in ("hello", "hi", "hey")):
        return f"Hello{', ' + name if name else ''}! It's good to talk with you."
    if "favorite color" in lower:
        return "I like mint green—it feels energetic and friendly. What color do you like?"
    if "thank" in lower:
        return f"You're welcome{', ' + name if name else ''}!"
    if any(word in lower for word in ("bye", "goodbye", "see you")):
        return f"Goodbye{', ' + name if name else ''}! Raise your hand whenever you'd like to talk again."
    if lower.endswith("?"):
        return "That's a thoughtful question. Demo mode uses local scripted replies, but I am still keeping this conversation separate from everyone else's."
    return f"I heard you say, “{text[:120]}.” In demo mode I can remember our conversation and keep it separate from the other people here."

