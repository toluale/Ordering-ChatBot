import os
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, AsyncGenerator, Dict, List, Union
from openai import AzureOpenAI
from openai.types.chat import (
    ChatCompletionSystemMessageParam,
    ChatCompletionUserMessageParam,
    ChatCompletionAssistantMessageParam
)
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions.kernel_arguments import KernelArguments

from streaming_ordering_chatbot.api.models import Message
from streaming_ordering_chatbot.api.content_safety import wrap_content_safety

# Load environment variables
load_dotenv()
# Azure OpenAI configuration
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

def get_required_env_var(name: str) -> str:
    """Get a required environment variable or raise an informative error."""
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"{name} environment variable is not set. "
            "Please set it in your .env file."
        )
    return value

# Set up logging
logger = logging.getLogger(__name__)

# Set up logging
logger = logging.getLogger(__name__)

class ConversationPlugin:
    """Base plugin for conversation handling with native SK functions."""
    
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
    
    @kernel_function(description="Prepares chat context for prompt template", name="chat")
    async def prepare_chat_context(self, chat_history: list[Message]) -> str:
        """Formats chat history for prompt template."""
        # Format chat history as string
        chat_str = "\n".join([f"{msg.role}: {msg.content}" for msg in chat_history])
        return f"Chat History:\n{chat_str}"


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
    """Base class for conversation flows using Semantic Kernel."""
    PROMPT_PATH = None
    PLUGIN_NAME = "conversation"
    MAX_TOKENS = 1000

    def __init__(self, endpoint: str, api_key: str, deployment_name: str, brand_name: Optional[str] = None):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
        
        # Initialize Semantic Kernel
        self.kernel = Kernel()
        self.chat_service = AzureChatCompletion(
            deployment_name=deployment_name,
            endpoint=endpoint,
            api_key=api_key,
            service_id="azurechat"
        )
        self.kernel.add_service(self.chat_service)
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            api_key=api_key,
            api_version="2024-11-20",
            azure_endpoint=endpoint
        )
        
        # Set up plugins
        self.conversation_plugin = ConversationPlugin(self.kernel)
        self.kernel.add_plugin(self.conversation_plugin, "conversation")
        
        # Add brand personality plugin
        from .brand_personality import BrandPersonalityPlugin
        self.brand_plugin = BrandPersonalityPlugin(self.kernel, brand_name)
        self.kernel.add_plugin(self.brand_plugin, "brand")
          # Load prompt template
        if self.PROMPT_PATH:
            self._setup_prompt_function()

    def _setup_prompt_function(self) -> None:
        """Set up the prompt function with brand personality integration."""
        try:
            if not self.PROMPT_PATH:
                raise ValueError("PROMPT_PATH must be set in the derived class")
                
            with open(self.PROMPT_PATH, "r", encoding="utf-8") as f:
                base_template = f.read()
            
            # Get brand instructions and combine them with the base template
            if hasattr(self, 'brand_plugin'):
                brand_instructions = self.brand_plugin.get_brand_instructions()
                prompt_template = f"{brand_instructions}\n\n{base_template}"
            else:
                prompt_template = base_template
            
            prompt_config = PromptTemplateConfig(
                template=prompt_template,
                name="chat",
                description="Chat with the assistant using brand personality"
            )
            
            self.kernel.add_function(
                plugin_name=self.PLUGIN_NAME,
                function_name="chat",
                prompt_template_config=prompt_config
            )
            
            logger.info("Successfully set up prompt function with brand personality")
        except Exception as e:
            logger.error(f"Error setting up prompt function: {e}")
            raise

    async def enhance_prompt_with_brand(self, system_prompt: str) -> str:
        try:
            if not hasattr(self, 'brand_plugin'):
                return system_prompt
                
            args = KernelArguments()
            args["system_prompt"] = system_prompt
            result = await self.kernel.invoke(
                plugin_name="brand",
                function_name="enhance_system_prompt",                arguments=args
            )
            return str(result) if result is not None else system_prompt
        except Exception:
            logger.warning("Failed to enhance prompt with brand personality", exc_info=True)
            return system_prompt

    async def _process_stream(
        self,
        completion,
        delay: float = 0.05
    ) -> AsyncGenerator[str, None]:
        try:
            current_word = []
            last_was_punctuation = False
            last_was_space = False

            for chunk in completion:
                if delay > 0:
                    await asyncio.sleep(delay)
                
                if (not chunk or not chunk.choices 
                    or not chunk.choices[0].delta 
                    or not hasattr(chunk.choices[0].delta, 'content')
                    or chunk.choices[0].delta.content is None):
                    continue
                
                token = chunk.choices[0].delta.content
                
                # token handling
                if token.strip() == "":  # All types of whitespace
                    if current_word:
                        yield "".join(current_word)
                        current_word = []
                    if not last_was_space and token != "\n":  # Only yield space if not after punctuation
                        yield " "
                        last_was_space = True
                    elif token == "\n":  # Always yield newlines
                        yield "\n"
                        last_was_space = True
                    last_was_punctuation = False
                elif token in ".!?,;:":  # Punctuation
                    if current_word:
                        yield "".join(current_word)
                        current_word = []
                    yield token
                    last_was_punctuation = True
                    last_was_space = False
                else:  # Regular text
                    if last_was_punctuation:
                        yield " "  # Add space after punctuation
                    current_word.append(token)
                    last_was_punctuation = False
                    last_was_space = False
            
            # Yield any remaining content
            if current_word:
                yield "".join(current_word)
                
        except Exception as e:
            logger.error(f"Error in stream processing: {e}")
            yield f"\nError: {str(e)}"

    @wrap_content_safety
    async def __call__(
        self,
        chat_history: List[Message],
        current_order: Optional[Dict] = None,
        delay: float = 0.05,
        model_deployment: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        current_order = current_order if current_order is not None else {"items": []}
        try:
            # Format chat history and order for the model
            context_args = KernelArguments()
            context_args["chat_history"] = chat_history
            context_args["current_order"] = current_order

            # Get system prompt from template and enhance with brand personality
            system_prompt = await self.kernel.invoke(
                plugin_name=self.PLUGIN_NAME,
                function_name="chat",
                arguments=context_args
            )
            enhanced_prompt = await self.enhance_prompt_with_brand(str(system_prompt))

            # Create messages with proper types and create messages list for completion
            messages: List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam]] = [
                ChatCompletionSystemMessageParam(role="system", content=str(enhanced_prompt))
            ]
            
            # Add conversation history
            for msg in chat_history:
                if msg.role == "user":
                    messages.append(ChatCompletionUserMessageParam(role="user", content=msg.content))
                elif msg.role == "assistant":
                    messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=msg.content))

            # Create completion stream
            completion = self.client.chat.completions.create(
                model=model_deployment or self.deployment_name,
                messages=messages,
                temperature=0.7,
                top_p=0.95,
                max_tokens=self.MAX_TOKENS,
                stream=True
            )

            # Stream tokens with proper formatting
            async for token in self._process_stream(completion, delay):
                yield token

        except Exception as e:
            error_msg = f"Error in conversation: {str(e)}"
            logger.error(error_msg)
            yield f"\nI apologize, but I encountered an error. Please try again."


# Specific conversation flows
class PreambleFlowSK(ConversationFlowSK):
    """Initial greeting and menu introduction flow."""
    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/preamble_SK.prompty")
    PLUGIN_NAME = "preamble"

    def __init__(self, endpoint: str, api_key: str, deployment_name: str, brand_name: Optional[str] = None):
        super().__init__(endpoint, api_key, deployment_name, brand_name)


class OrderAssistantFlowSK(ConversationFlowSK):
    """Main order taking and menu assistance flow."""
    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/order_intent_SK.prompty")
    PLUGIN_NAME = "order"

    def __init__(self, endpoint: str, api_key: str, deployment_name: str, brand_name: Optional[str] = None):
        super().__init__(endpoint, api_key, deployment_name, brand_name)
        # Add OrderPlugin for menu and order context
        self.order_plugin = OrderPlugin(self.kernel)
        self.kernel.add_plugin(self.order_plugin, "order")


class SummaryFlowSK(ConversationFlowSK):
    """Order summary and confirmation flow."""
    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/summary_SK.prompty")
    PLUGIN_NAME = "summary"

    def __init__(self, endpoint: str, api_key: str, deployment_name: str, brand_name: Optional[str] = None):
        super().__init__(endpoint, api_key, deployment_name, brand_name)
