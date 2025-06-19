import asyncio
from copy import copy
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional

from promptflow.core import AzureOpenAIModelConfiguration, Prompty
from promptflow.tracing import trace

from streaming_ordering_chatbot.api.content_safety import wrap_content_safety
from streaming_ordering_chatbot.api.flows.schemas import LLMOrder
from streaming_ordering_chatbot.api.models import Message

handler = logging.FileHandler("streaming_ordering_chatbot.conversation_flow.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ConversationFlow:
    PROMPT_PATH = None

    def __init__(self, model_config: AzureOpenAIModelConfiguration):
        """Base class for conversation flows that return streaming responses.

        Args:
            model_config (AzureOpenAIModelConfiguration): Model configuration overrides.
        """
        self.model_config = model_config
        self.model = {
            "configuration": self.model_config,
            "parameters": {
                "max_tokens": 1000,
                "stream": True,
            },
            "response": "all",
        }

    @wrap_content_safety
    @trace
    async def __call__(
        self,
        chat_history: list[Message],
        delay: float = 0.05,
        personality: Optional[str] = None,
        model_deployment: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Executes LLM inference and streams response

        Args:
            chat_history (list[Message]): Chat history.
            delay (float, optional): Token delay for streaming demonstration. Defaults to 0.05.

        Yields:
            AsyncGenerator[str]: Streaming token response
        """

        overrides = {}
        if model_deployment:
            overrides["configuration"] = copy(self.model["configuration"])
            overrides["configuration"].azure_deployment = model_deployment

        generator = Prompty.load(
            source=self.PROMPT_PATH, model={**self.model, **overrides}
        )
        response = generator(chat_history=chat_history, personality=personality)

        for chunk in response:
            if chunk and len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content
                if content:
                    logger.info("Produced token: %s", content)
                    yield content + "\n"
            await asyncio.sleep(delay)


class PreambleFlow(ConversationFlow):
    """
    Flow class for handling the preamble conversation.

    Attributes:
        PROMPT_PATH (Path): The path to the preamble prompty file.
    """

    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/preamble.prompty")


class SummaryFlow(ConversationFlow):
    """
    SummaryFlow is a class for generating a summary.

    Attributes:
        PROMPT_PATH (Path): The path to the summary prompty file.

    """

    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/summary.prompty")


class OrderAssistantFlow(ConversationFlow):
    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/assistant.prompty")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_prompts()

    def _load_prompts(self):
        """Load static menu content"""
        with Path(__file__).parent.joinpath("prompts/menu.txt").open() as f:
            self.menu = f.read()

    @wrap_content_safety
    async def __call__(
        self,
        chat_history: list[Message],
        current_order: dict,
        delay: float = 0.05,
        personality: Optional[str] = None,
        model_deployment: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Executes LLM inference and streams response

        Args:
            chat_history (list[Message]): Chat history.
            current_order (dict): Current LLM order.
            delay (float, optional): Token delay for streaming demonstration. Defaults to 0.05.

        Yields:
            AsyncGenerator[str]: Streaming token response
        """
        overrides = {}
        if model_deployment:
            overrides["configuration"] = copy(self.model["configuration"])
            overrides["configuration"].azure_deployment = model_deployment

        current_order = LLMOrder.model_validate(current_order).model_dump()

        response_generator = Prompty.load(
            source=self.PROMPT_PATH, model={**self.model, **overrides}
        )
        response = response_generator(
            chat_history=chat_history,
            current_order=current_order,
            menu=self.menu,
            personality=personality,
        )

        for chunk in response:
            if chunk and len(chunk.choices) > 0:
                content = chunk.choices[0].delta.content
                if content:
                    yield content + "\n"
            await asyncio.sleep(delay)
