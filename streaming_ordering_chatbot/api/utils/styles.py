from typing import List
from enum import Enum


class ConversationStyleEnum(str, Enum):
    DEFAULT = "default"
    CASUAL = "casual"
    GENZ = "genz"


def valid_styles() -> List[str]:
    return [e.value for e in ConversationStyleEnum]


def parse_style(style: str | None, default: str = ConversationStyleEnum.DEFAULT.value) -> str:
    if not style:
        return default
    style_lower = style.lower()
    return style_lower if style_lower in valid_styles() else default
