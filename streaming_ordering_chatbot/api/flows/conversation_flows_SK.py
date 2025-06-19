import asyncio
from copy import copy
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, Any

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments

from streaming_ordering_chatbot.api.content_safety import wrap_content_safety
from streaming_ordering_chatbot.api.flows.schemas import LLMOrder
from streaming_ordering_chatbot.api.models import Message
from streaming_ordering_chatbot.api.flows.brand_personality import BrandPersonalityPlugin

# Set up logging
handler = logging.FileHandler("streaming_ordering_chatbot.conversation_flow_sk.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


class ConversationPlugin:
    """Base plugin for conversation handling with native SK functions."""
    
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
    
    @kernel_function(description="Prepares chat context for prompt template", name="chat")
    async def prepare_chat_context(self, chat_history: list[Message]) -> str:
        """Formats chat history for prompt template."""
        # Format chat history as string
        chat_str = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
        return f"Chat History:\n{chat_str}"    @kernel_function(description="Formats response for streaming", name="format_response")
    async def format_response(self, response: str) -> str:
        """Returns the response as-is, relying on the model's formatting."""
        return response


class OrderPlugin(ConversationPlugin):
    """Plugin for order-related conversation functions."""
    
    def __init__(self, kernel: Kernel):
        super().__init__(kernel)
        self.menu_path = Path(__file__).parent.joinpath("prompts/menu.txt")
        with open(self.menu_path, "r") as f:
            self.menu = f.read()
    
    @kernel_function(description="Prepares order context for prompt template", name="chat")
    async def prepare_order_context(self, chat_history: list[Message], current_order: dict) -> str:
        """Formats order context for prompt template."""
        # Convert objects to their string representations for the prompt
        chat_str = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
        order_str = "\n".join([f"{k}: {v}" for k, v in current_order.items()])
        
        # Format the context in a consistent way
        context = f"Chat History:\n{chat_str}\n\nCurrent Order:\n{order_str}\n\nMenu:\n{self.menu}"
        return context


class ConversationFlowSK:
    PROMPT_PATH = None
    PLUGIN_NAME = "conversation"
    MAX_TOKENS = 1000  # Add token limit

    def __init__(self, endpoint: str, api_key: str, deployment_name: str):
        """Base class for conversation flows using Semantic Kernel.

        Args:
            endpoint (str): Azure OpenAI endpoint
            api_key (str): Azure OpenAI API key
            deployment_name (str): Model deployment name
        """
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
          # Initialize Semantic Kernel
        self.kernel = Kernel()
        chat_service = AzureChatCompletion(
            deployment_name=self.deployment_name,
            endpoint=self.endpoint,
            api_key=self.api_key,
            service_id="azurechat"
        )
        # Add the chat service to the kernel
        self.kernel.add_service(chat_service)
        
        # Set up basic conversation plugin
        self.conversation_plugin = ConversationPlugin(self.kernel)
        self.kernel.add_plugin(self.conversation_plugin, "conversation")
          # Load and register prompt template
        if self.PROMPT_PATH:
            self._setup_prompt_function()
            
    def _setup_prompt_function(self):
        """Load and register the prompt template as a semantic function"""
        try:
            if self.PROMPT_PATH is None:
                raise ValueError("PROMPT_PATH must be set in the derived class")
                
            with open(self.PROMPT_PATH, "r", encoding="utf-8") as f:
                prompt_template = f.read()
            
            prompt_config = PromptTemplateConfig(
                name="chat",
                description="Chat with the assistant",
                template=prompt_template
            )
            
            self.kernel.add_function(
                plugin_name=self.PLUGIN_NAME,
                function_name="chat",
                prompt_template_config=prompt_config
            )
        except Exception as e:
            logger.error(f"Error setting up prompt function: {e}")
            raise

    @wrap_content_safety
    async def __call__(
        self,
        chat_history: list[Message],
        delay: float = 0.05,
        model_deployment: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Executes LLM inference and streams response

        Args:
            chat_history (list[Message]): Chat history
            delay (float, optional): Token delay for streaming. Defaults to 0.05.
            model_deployment (Optional[str], optional): Override model deployment. Defaults to None.

        Yields:
            AsyncGenerator[str, None]: Streaming token response
        """
        original_deployment = None
        
        try:
            if model_deployment:
                original_deployment = self.kernel.get_service("azurechat").deployment_name
                self.kernel.get_service("azurechat").deployment_name = model_deployment
            
            # Pass chat_history directly to match prompt template variable name
            args = KernelArguments(chat_history=chat_history)

            # Get streaming response
            generator = self.kernel.invoke_stream(
                plugin_name=self.PLUGIN_NAME,
                function_name="chat",
                arguments=args
            )

            # Stream the response
            async for response in generator:
                if isinstance(response, list):
                    for chunk in response:
                        text = str(chunk)
                        if text:
                            logger.info("Produced token: %s", text)
                            yield text.strip()
                            await asyncio.sleep(delay)
                elif response:
                    text = str(response)
                    if text:
                        logger.info("Produced token: %s", text)
                        yield text.strip()
                        await asyncio.sleep(delay)

        finally:
            if original_deployment:
                self.kernel.get_service("azurechat").deployment_name = original_deployment


class PreambleFlowSK(ConversationFlowSK):
    """Flow class for handling the preamble conversation using Semantic Kernel."""
    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/preamble_SK.prompty")
    PLUGIN_NAME = "preamble"


class SummaryFlowSK(ConversationFlowSK):
    """Flow class for generating conversation summaries using Semantic Kernel."""
    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/summary_SK.prompty")
    PLUGIN_NAME = "summary"


class OrderAssistantFlowSK(ConversationFlowSK):
    """Flow class for order-related conversations using Semantic Kernel."""
    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/assistant_SK.prompty")
    PLUGIN_NAME = "assistant"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._load_prompts()        # Set up order plugin
        self.order_plugin = OrderPlugin(self.kernel)
        self.kernel.add_plugin(self.order_plugin, "order")

    def _load_prompts(self):
        """Load static menu content"""
        try:
            with Path(__file__).parent.joinpath("prompts/menu.txt").open() as f:
                self.menu = f.read()
        except Exception as e:
            logger.error(f"Error loading menu: {e}")
            self.menu = "Error loading menu content"

    @wrap_content_safety
    async def __call__(
        self,
        chat_history: list[Message],
        current_order: Optional[dict] = None,
        delay: float = 0.05,
        model_deployment: Optional[str] = None,
    ) -> AsyncGenerator[str, None]:
        """Executes LLM inference and streams response

        Args:
            chat_history (list[Message]): Chat history
            current_order (Optional[dict]): Current order state. Defaults to None.
            delay (float, optional): Token delay for streaming. Defaults to 0.05.
            model_deployment (Optional[str], optional): Override model deployment. Defaults to None.

        Yields:
            AsyncGenerator[str, None]: Streaming token response
        """
        original_deployment = None
        
        try:
            if model_deployment:
                original_deployment = self.kernel.get_service("azurechat").deployment_name
                self.kernel.get_service("azurechat").deployment_name = model_deployment
            
            # Set up kernel arguments with both chat history and order
            args = KernelArguments(
                chat_history=chat_history,
                current_order=current_order or {"items": []},
                menu=self.menu
            )

            # Get streaming response
            generator = self.kernel.invoke_stream(
                plugin_name=self.PLUGIN_NAME,
                function_name="chat",
                arguments=args
            )

            # Stream the response
            async for response in generator:
                if isinstance(response, list):
                    for chunk in response:
                        text = str(chunk)
                        if text:
                            logger.info("Produced token: %s", text)
                            yield text.strip()
                            await asyncio.sleep(delay)
                elif response:
                    text = str(response)
                    if text:
                        logger.info("Produced token: %s", text)
                        yield text.strip()
                        await asyncio.sleep(delay)

        finally:
            if original_deployment:
                self.kernel.get_service("azurechat").deployment_name = original_deployment
