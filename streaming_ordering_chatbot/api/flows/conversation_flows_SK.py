import asyncio
from copy import copy
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional, Dict, Any, List

from openai import AzureOpenAI
from openai.types.chat import ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion, OpenAIPromptExecutionSettings
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent
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

# Type for chat messages
ChatMessage = Dict[str, str]


class ConversationPlugin:
    """Base plugin for conversation handling with native SK functions."""
    
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
    
    @kernel_function(description="Prepares chat context for prompt template", name="chat")
    async def prepare_chat_context(self, chat_history: list[Message]) -> str:
        """Formats chat history for prompt template."""
        # Format chat history as string
        chat_str = "\n".join([f"{msg.role}: { msg.content }" for msg in chat_history])
        return f"Chat History:\n{chat_str}"    
    
    @kernel_function(description="Formats response for streaming", name="format_response")
    async def format_response(self, response: str) -> str:
        """Formats the response with proper spacing between words, sentences, and after punctuation.
        
        Args:
            response (str): The raw response string to format
            
        Returns:
            str: The formatted response with proper spacing
        """
        if not response:
            return ""
            
        # First handle basic word splitting
        words = []
        current_word = ""
        
        for i, char in enumerate(response):
            if char.isupper() and i > 0:
                # If we see an uppercase letter and it's not the start,
                # it likely indicates a new word
                if current_word:
                    words.append(current_word)
                current_word = char
            else:
                current_word += char
                
            # Handle bullet points
            if char == '*':
                if current_word:
                    words.append(current_word[:-1])  # Remove the * from previous word
                words.append('*')
                current_word = ""
                
        if current_word:
            words.append(current_word)
            
        # Join words with spaces
        formatted = ' '.join(word for word in words if word)
        
        # Handle punctuation spacing
        for punct in '.!?,':
            formatted = formatted.replace(f' {punct}', punct)  # Remove space before punctuation
            formatted = formatted.replace(f'{punct}', f'{punct} ')  # Add space after punctuation
            
        # Clean up multiple spaces and handle special characters
        formatted = ' '.join(formatted.split())
        
        # Special handling for bullet points
        formatted = formatted.replace('* ', '\n* ')
        
        return formatted.strip()


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
        chat_str = "\n".join([f"{msg.role}: { msg.content }" for msg in chat_history])
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
        
        # Initialize Semantic Kernel for prompt handling
        self.kernel = Kernel()
        self.chat_service = AzureChatCompletion(
            deployment_name=self.deployment_name,
            endpoint=self.endpoint,
            api_key=self.api_key,
            service_id="azurechat"
        )
        # Add the chat service to the kernel
        self.kernel.add_service(self.chat_service)
          # Initialize Azure OpenAI client for streaming
        self.client = AzureOpenAI(
            api_key=self.api_key,
            api_version="2024-02-15-preview",
            azure_endpoint=self.endpoint
        )
        
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
                template=prompt_template,
                name="chat",
                description="Chat with the assistant"
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
        """Execute LLM inference and stream response using direct Azure OpenAI client.
        
        Args:
            chat_history: List of chat messages
            delay: Delay between yielding tokens
            model_deployment: Optional model deployment override
            
        Yields:
            Tokens from the streaming response
        """
        try:
            # Format chat history for the model
            args = KernelArguments(chat_history=chat_history)
            
            # Get system prompt from template
            system_prompt = await self.kernel.invoke(
                plugin_name=self.PLUGIN_NAME,
                function_name="chat",
                arguments=args
            )
            
            # Create messages with proper types
            messages = (
                [ChatCompletionSystemMessageParam(role="system", content=str(system_prompt))] +
                [
                    ChatCompletionUserMessageParam(role="user", content=msg.content)
                    if msg.role == "user"
                    else ChatCompletionAssistantMessageParam(role="assistant", content=msg.content)
                    for msg in chat_history
                    if msg.role in ["user", "assistant"]
                ]
            )
            
            # Get streaming response using direct Azure OpenAI client
            completion = self.client.chat.completions.create(
                model=model_deployment or self.deployment_name,
                messages=messages,
                temperature=0.7,
                top_p=0.95,
                max_tokens=self.MAX_TOKENS,
                stream=True
            )
            
            # Stream the response chunks
            previous_text = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    text = chunk.choices[0].delta.content
                    # Insert a space if needed
                    if previous_text and not previous_text.endswith(" ") and not text.startswith(" "):
                        logger.info("Inserted space between tokens")
                        yield " "
                        await asyncio.sleep(delay)
                    logger.info("Produced token: %s", text)
                    yield text
                    previous_text = text
                    await asyncio.sleep(delay)
                        
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise


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
        """Execute LLM inference and stream response using direct Azure OpenAI client.
        
        Args:
            chat_history: List of chat messages
            current_order: Current order state
            delay: Delay between yielding tokens
            model_deployment: Optional model deployment override
            
        Yields:
            Tokens from the streaming response
        """
        try:
            # Set up kernel arguments with both chat history and order
            args = KernelArguments(
                chat_history=chat_history,
                current_order=current_order or {"items": []},
                menu=self.menu
            )
            
            # Get system prompt from template
            system_prompt = await self.kernel.invoke(
                plugin_name=self.PLUGIN_NAME,
                function_name="chat",
                arguments=args
            )
            
            # Create messages with proper types
            messages = (
                [ChatCompletionSystemMessageParam(role="system", content=str(system_prompt))] +
                [
                    ChatCompletionUserMessageParam(role="user", content=msg.content)
                    if msg.role == "user"
                    else ChatCompletionAssistantMessageParam(role="assistant", content=msg.content)
                    for msg in chat_history
                    if msg.role in ["user", "assistant"]
                ]
            )
            
            # Get streaming response using direct Azure OpenAI client
            completion = self.client.chat.completions.create(
                model=model_deployment or self.deployment_name,
                messages=messages,
                temperature=0.7,
                top_p=0.95,
                max_tokens=self.MAX_TOKENS,
                stream=True
            )
            
            # Stream the response chunks
            previous_text = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    text = chunk.choices[0].delta.content
                    # Insert a space if needed
                    if previous_text and not previous_text.endswith(" ") and not text.startswith(" "):
                        logger.info("Inserted space between tokens")
                        yield " "
                        await asyncio.sleep(delay)
                    logger.info("Produced token: %s", text)
                    yield text
                    previous_text = text
                    await asyncio.sleep(delay)
                        
        except Exception as e:
            logger.error(f"Error in chat completion: {e}")
            raise
