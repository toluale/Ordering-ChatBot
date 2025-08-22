from typing import List

# Markers that indicate prompt context sections we should strip from assistant echoes
CONTEXT_MARKERS: List[str] = [
    "Previous conversation:",
    "Current Order:",
    "Available menu:",
    "Current order status:",
    "Menu:",
    "Chat History:",
    "Instructions:",
    "Reference Information",
    "Brand:",
    "[CONTEXT]",
    "[END CONTEXT]",
    "MENU",
    "CURRENT ORDER",
    "CHAT HISTORY",
    "CONVERSATION HISTORY",
]


def clean_assistant_response(content: str) -> str:
    """Remove known context markers from assistant responses to keep only the reply.

    This is used to prevent the model from echoing the prompt context back to the user.
    """
    if not content:
        return ""

    cleaned = content
    for marker in CONTEXT_MARKERS:
        if marker in cleaned:
            cleaned = cleaned.split(marker)[0].strip()
    return cleaned
