import asyncio
import json
import logging
from typing import AsyncGenerator, Awaitable, Callable, Optional

logger = logging.getLogger(__name__)


def _get_delta_arguments(chunk) -> Optional[str]:
    if not getattr(chunk, "choices", None):
        return None
    if not chunk.choices:
        return None
    delta = chunk.choices[0].delta
    tool_calls = getattr(delta, "tool_calls", None)
    if not tool_calls:
        return None
    try:
        func = getattr(tool_calls[0], "function", None)
        args = getattr(func, "arguments", None) if func else None
        return args if isinstance(args, str) else None
    except (AttributeError, IndexError):
        return None


def _maybe_parse_items_from_buffer(json_buffer: str) -> Optional[dict]:
    if "}" not in json_buffer:
        return None
    parse_string = json_buffer[: json_buffer.rfind("}") + 1]
    left_brace = parse_string.count("{")
    right_brace = parse_string.count("}")
    if right_brace > 0 and (left_brace - right_brace) == 1:
        try:
            return json.loads(parse_string + "]}")
        except json.JSONDecodeError:
            return None
    return None


async def process_order_function_stream(
    response,
    validate_item: Callable[[dict], Awaitable[object | None]],
    finalize_order: Callable[[dict], str],
    delay: float = 0.05,
) -> AsyncGenerator[str, None]:
    """Process Azure OpenAI function-call streaming for order items.

    Parameters:
      - response: streaming response iterator from Azure OpenAI
      - validate_item: async callable that validates a candidate item dict and returns a model or None
      - finalize_order: callable that takes the accumulated order items dict and returns a final JSON string
      - delay: optional sleep between chunks for smoother UI
    Yields:
      - JSON lines containing the progressively built order
    """
    json_string = ""
    current_order_items: dict = {"items": []}
    prev_item_id = -1
    first_item = True

    try:
        for chunk in response:
            if delay > 0:
                await asyncio.sleep(delay)

            arguments = _get_delta_arguments(chunk)
            if not arguments:
                continue

            json_string += arguments

            maybe_state = _maybe_parse_items_from_buffer(json_string)
            if not maybe_state:
                continue

            current_order_items = maybe_state
            items = current_order_items.get("items") or []
            if not items:
                continue

            latest_item = items[-1]
            current_item_id = latest_item.get("itemId")
            if current_item_id == prev_item_id:
                continue

            prev_item_id = current_item_id
            validated_item = await validate_item(latest_item)
            if validated_item is None:
                logger.warning("Item validation failed for: %s", latest_item)
                continue

            ser_item = getattr(validated_item, "model_dump", lambda: validated_item)()
            if first_item:
                first_item = False
                yield '{"order": [' + json.dumps(ser_item) + "\n"
            else:
                yield "," + json.dumps(ser_item) + "\n"

        if not first_item:
            yield "]}\n"

        # Always emit final order structure using provided finalizer
        yield finalize_order(current_order_items)
    except Exception as e:
        logger.error("Error in process_order_function_stream: %s", e)
        yield json.dumps({"error": str(e)}) + "\n"
