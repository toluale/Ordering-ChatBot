import asyncio
import logging
from typing import AsyncGenerator, Optional

logger = logging.getLogger(__name__)


def _extract_token_from_chunk(chunk) -> Optional[str]:
    if (
        not chunk
        or not getattr(chunk, "choices", None)
        or not chunk.choices
        or not getattr(chunk.choices[0], "delta", None)
        or not hasattr(chunk.choices[0].delta, "content")
        or chunk.choices[0].delta.content is None
    ):
        return None
    return chunk.choices[0].delta.content


def _is_header_incomplete(text: str) -> bool:
    return text.strip().startswith("#") and not text.endswith("\n")


def _inside_unclosed_markdown(text: str) -> bool:
    open_bold = text.count("**") % 2
    open_italic = text.count("*") % 2
    return (open_bold != 0) or (open_italic != 0)


def _should_yield_on_token(token: str, buffer: list[str], buffer_text: str) -> bool:
    if token in ".!?\n":
        return True
    if token == "\n" and len(buffer) > 1 and buffer[-2] == "\n":
        return True
    if token == "\n" and buffer_text.strip().startswith("#"):
        return True
    if len(buffer) > 50:
        return True
    return False


async def process_chat_stream(
    completion,
    delay: float = 0.05,
) -> AsyncGenerator[str, None]:
    """Process Azure OpenAI streaming chat completion into sensible chunks.

    - Preserves markdown structure (headers, bold/italic) where possible
    - Yields on sentence boundaries and paragraph breaks
    - Adds an optional small delay for smoother UI streaming
    """
    try:
        buffer: list[str] = []

        for chunk in completion:
            if delay > 0:
                await asyncio.sleep(delay)

            token = _extract_token_from_chunk(chunk)
            if token is None:
                continue

            buffer.append(token)
            buffer_text = "".join(buffer)

            # Avoid breaking markdown headers or inline formatting
            if _is_header_incomplete(buffer_text):
                continue
            if _inside_unclosed_markdown(buffer_text) and token not in ["\n", ".", "!", "?"]:
                continue

            if _should_yield_on_token(token, buffer, buffer_text):
                text = "".join(buffer)
                if text:
                    yield text
                buffer = []

        # Yield any remaining content
        if buffer:
            final_text = "".join(buffer)
            if final_text:
                yield final_text

    except Exception as e:
        logger.error(f"Error in stream processing: {e}")
        yield f"\nError: {str(e)}"
