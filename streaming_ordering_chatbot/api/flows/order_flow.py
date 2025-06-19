import asyncio
import json
import logging
from copy import copy
from pathlib import Path
from typing import AsyncGenerator, Union, Optional

from promptflow.core import AzureOpenAIModelConfiguration, Prompty
from promptflow.tracing import trace

from streaming_ordering_chatbot.api.flows.schemas import (
    LLMBurgerItem,
    LLMDrinkItem,
    LLMFriesItem,
    LLMOrder,
)
from streaming_ordering_chatbot.api.models import Message

handler = logging.FileHandler("streaming_ordering_chatbot.order_flow.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class OrderFlow:
    def __init__(self, model_config: AzureOpenAIModelConfiguration):
        """Flow initialization

        Args:
            model_config (AzureOpenAIModelConfiguration): Model configuration overrides.
        """
        self.model_config = model_config
        self._load_prompts()
        self.model = {
            "configuration": self.model_config,
            "parameters": {"max_tokens": 1000, "stream": True, **self.tools},
            "response": "all",
        }

    def _load_prompts(self):
        """Load static prompt content"""
        with Path(__file__).parent.joinpath("prompts/tools.json").open() as f:
            self.tools = json.load(f)

        with Path(__file__).parent.joinpath("prompts/menu.txt").open() as f:
            self.menu = f.read()

    @staticmethod
    async def _validate_item(
        item: dict,
    ) -> Union[None, LLMBurgerItem, LLMDrinkItem, LLMFriesItem]:
        """Validates an item and returns a valid item if successful

        Args:
            item (dict): Item to validate

        Returns:
            Union[None, LLMBurgerItem, LLMDrinkItem, LLMFriesItem]:
        """
        try:
            # Attempt validation
            items = LLMOrder.model_validate({"items": [item]}).items
            if not items:
                logger.warning("Invalid item: %s", item)
                return None
            val_item = items[0].to_order_item()
        except ValueError as e:
            logger.error("Error validating item: %s", e)
            return None
        return val_item

    @trace
    async def __call__(
        self,
        chat_history: list[Message],
        current_order: dict,
        delay: float = 0.05,
        model_deployment: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Executes LLM call and emits validated items

        Args:
            chat_history (list[Message]): Chat history
            current_order (dict): Current order as a dictionary
            delay (float, optional): Token generation delay for streaming demonstration. Defaults to 0.05.

        Yields:
            AsyncGenerator[str]: validated order items
        """

        overrides = {}
        if model_deployment:
            overrides["configuration"] = copy(self.model["configuration"])
            overrides["configuration"].azure_deployment = model_deployment

        current_order = LLMOrder.model_validate(current_order).model_dump()

        order_generator = Prompty.load(
            source=Path(__file__).parent.joinpath("prompts/order.prompty"),
            model={**self.model, **overrides},
        )

        response = order_generator(
            chat_history=chat_history, current_order=current_order, menu=self.menu
        )

        json_string = ""
        current_order = {"items": []}
        validated_items = []
        prev_item_id = -1
        first_item = True
        for chunk in response:
            await asyncio.sleep(delay)
            if (
                chunk is not None
                and hasattr(chunk, "choices")
                and len(chunk.choices) > 0
            ):
                try:
                    arg = chunk.choices[0].delta.tool_calls[0].function.arguments
                    logger.info("Produced token: %s", arg)
                    # First response states function_name then subsequent responses state arguments
                    # Ensure that a token is produced for args
                    if isinstance(arg, str):
                        json_string += arg
                        # Only want items that have been closed so last token must have a closing curly bracket
                        if "}" in arg:
                            # In case item is closed and new item opened in the same token
                            parse_string = json_string[: json_string.rfind("}") + 1]
                            left_brace = parse_string.count("{")
                            right_brace = parse_string.count("}")
                            # Only true if the last item is closed
                            if right_brace > 0 and (left_brace - right_brace) == 1:
                                # Parse partial json with missing closing square bracket and curly brace
                                current_order = json.loads(parse_string + "]}")
                                # Check that current item is different than previously parsed item
                                if (
                                    items := current_order.get("items")
                                ) is not None and items[-1].get(
                                    "itemId"
                                ) != prev_item_id:
                                    prev_item_id = items[-1].get("itemId")
                                    # Validate item against Order schema
                                    validated_item = await self._validate_item(
                                        items[-1]
                                    )
                                    if validated_item is not None:
                                        ser_item = validated_item.model_dump()
                                        if first_item:
                                            first_item = False
                                            yield '{"order": [' + json.dumps(
                                                ser_item
                                            ) + "\n"
                                        else:
                                            yield "," + json.dumps(ser_item) + "\n"
                                        validated_items.append(validated_item)
                except TypeError as e:
                    # Catch errors for missing tool_calls, choices
                    logger.error("Error occurred: %s", e)
        # Close `items` property
        yield "]}" + "\n"
        # Finally send back current LLMOrder for client side storage for subsequent requests
        yield json.dumps(
            {"LLMOrder": LLMOrder.model_validate(current_order).model_dump()}
        ) + "\n"
