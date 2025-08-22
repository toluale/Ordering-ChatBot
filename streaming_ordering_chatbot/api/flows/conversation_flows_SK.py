import os
import asyncio
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, AsyncGenerator, Dict, List, Union, Any
from openai import AzureOpenAI
from openai.types.chat import (ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam)
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions.kernel_arguments import KernelArguments

from streaming_ordering_chatbot.api.models import Message
from streaming_ordering_chatbot.api.content_safety import wrap_content_safety
from streaming_ordering_chatbot.api.utils.text import clean_assistant_response
from streaming_ordering_chatbot.api.utils.azure_client import create_azure_openai_client, build_chat_params
from streaming_ordering_chatbot.api.utils.stream_utils import process_chat_stream
from streaming_ordering_chatbot.api.utils.prompt_utils import build_overlay_cache_key, make_overlay_parts, enhance_prompt_with_parts

# Load environment variables
load_dotenv()
# Azure OpenAI configuration
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
'''
def get_required_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"{name} environment variable is not set. Please set it in your .env file."
        )
    return value
'''
# Set up logging
logger = logging.getLogger(__name__)


"""Centralized assistant response cleaning now lives in utils.text.clean_assistant_response."""


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
    """Plugin for order-related conversation functions.
    """
    
    def __init__(self, kernel: Kernel):
        super().__init__(kernel)
        # Import MenuManager for menu loading
        from .menu_manager import MenuManager
        self.menu_manager = MenuManager()
        self.menu = self._load_menu_for_current_brand()
        
    def _load_menu_for_current_brand(self) -> str:
        """Load menu using MenuManager for the current brand (centralized via MenuManager)."""
        try:
            current_brand = self.menu_manager.require_current_brand()
            return self.menu_manager.get_menu_text_format(current_brand)
        except Exception as e:
            logger.error(f"Failed to load menu using MenuManager: {e}")
            raise RuntimeError(
                f"Menu loading failed: {e}. Please ensure RESTAURANT_BRAND is set and a corresponding menu file exists."
            ) from e
        
    @kernel_function(description="Prepares order context for prompt template", name="chat")
    async def prepare_order_context(self, chat_history: list[Message], current_order: dict, brand_name: Optional[str] = None) -> str:
        """Formats order context for prompt template."""
        # Clean and format chat history
        clean_history = []
        for msg in chat_history:
            content = msg.content
            if msg.role == "assistant":
                content = clean_assistant_response(content)
            clean_history.append(f"{msg.role}: {content}")
        
        # Get recent conversation history
        chat_str = "\n".join(clean_history[-8:])  # Keep last 8 messages for immediate context
        
        items = current_order.get("items", [])
        order_str = "Current items in order: " + ", ".join(str(item) for item in items) if items else "No items in order yet"
        
        # Get brand name from brand plugin if not provided
        if not brand_name:
            # Try to get current brand from menu manager first
            brand_name = self.menu_manager.get_current_brand()
            
            # If not available and kernel exists, try brand plugin
            if not brand_name and hasattr(self, 'kernel'):
                try:
                    # Try to get current brand from brand plugin
                    brand_result = await self.kernel.invoke(
                        plugin_name="brand",
                        function_name="get_current_brand"
                    )
                    brand_name = str(brand_result).split(":")[1].strip() if ":" in str(brand_result) else ""
                except:
                    pass
            
            # Final fallback to generic name if still not found
            brand_name = brand_name or "Restaurant"

        # Format context with clear section markers
        context = (
            f"Brand Name: {brand_name}\n\n"
            f"Previous Conversation:\n{chat_str}\n\n"
            f"Current Order Status:\n{order_str}\n\n"
            f"Available Menu:\n{self.menu}\n\n"
            "Instructions: Use the above information to assist the customer with their order. "
            "Keep track of items ordered and respond naturally to their requests."
        )
        return context


class ConversationFlowSK:
    """Base class for conversation flows using Semantic Kernel."""
    PROMPT_PATH = None
    PLUGIN_NAME = "conversation"
    MAX_TOKENS = 2500

    def __init__(self, ENDPOINT: str, API_KEY: str, DEPLOYMENT_NAME: str, BRAND_NAME: Optional[str] = None, CONVERSATION_STYLE: Optional[str] = None):
        self.ENDPOINT = ENDPOINT
        self.API_KEY = API_KEY
        self.DEPLOYMENT_NAME = DEPLOYMENT_NAME

        # Initialize Semantic Kernel
        self.kernel = Kernel()
        self.chat_service = AzureChatCompletion(
            deployment_name=self.DEPLOYMENT_NAME,
            endpoint=self.ENDPOINT,
            api_key=self.API_KEY,
            service_id="azurechat"
        )
        self.kernel.add_service(self.chat_service)

        # Initialize Azure OpenAI client
        self.client = create_azure_openai_client(api_key=self.API_KEY, endpoint=self.ENDPOINT)

        # Set up plugins
        self.conversation_plugin = ConversationPlugin(self.kernel)
        self.kernel.add_plugin(self.conversation_plugin, "conversation")

        # Add brand personality plugin
        from .brand_personality import BrandPersonalityPlugin
        self.brand_plugin = BrandPersonalityPlugin(self.kernel, BRAND_NAME)
        self.kernel.add_plugin(self.brand_plugin, "brand")

        # Store conversation style for use in prompt templates
        self.conversation_style = CONVERSATION_STYLE or "default"

        # Always add conversation style plugin (not conditional)
        from .conversation_style import ConversationStylePlugin, ConversationStyle
        try:
            style_enum = ConversationStyle(self.conversation_style.lower())
        except ValueError:
            style_enum = ConversationStyle.DEFAULT

        self.style_plugin = ConversationStylePlugin(self.kernel, style_enum)
        self.kernel.add_plugin(self.style_plugin, "style")
        logger.info(f"Added conversation style plugin: {self.conversation_style}")

        # Cache for brand/style overlays to avoid recomputing on each call
        self._overlay_cache = {}

        # Load prompt template
        if self.PROMPT_PATH:
            self._setup_prompt_function()

    def _setup_prompt_function(self) -> None:
        """Register the base (pure) template without brand/style prepends."""
        try:
            if not self.PROMPT_PATH:
                raise ValueError("PROMPT_PATH must be set in the derived class")
                
            with open(self.PROMPT_PATH, "r", encoding="utf-8") as f:
                base_template = f.read()

            # Keep templates pure; brand/style applied at runtime in enhance_prompt_with_brand_and_style
            prompt_template = base_template

            prompt_config = PromptTemplateConfig(
                template=prompt_template,
                name="chat",
                description="Chat with the assistant using brand personality and conversation style"
            )
            
            self.kernel.add_function(
                plugin_name=self.PLUGIN_NAME,
                function_name="chat",
                prompt_template_config=prompt_config
            )
            logger.info("Successfully set up prompt function with brand personality and conversation style")
        except Exception as e:
            logger.error(f"Error setting up prompt function: {e}")
            raise

    async def enhance_prompt_with_brand_and_style(self, system_prompt: str) -> str:
        """Enhance system prompt with both brand personality and conversation style with caching."""
        try:
            brand = getattr(self.brand_plugin, 'current_brand', None) if hasattr(self, 'brand_plugin') else None
            style_val = self.style_plugin.current_style.value if hasattr(self, 'style_plugin') and self.style_plugin is not None else "default"
            tmpl = str(self.PROMPT_PATH) if self.PROMPT_PATH else ""
            cache_key = build_overlay_cache_key(brand, style_val, tmpl)

            if cache_key not in self._overlay_cache:
                brand_instr = self.brand_plugin.get_brand_instructions() if hasattr(self, 'brand_plugin') and brand else None
                style_instr = self.style_plugin.get_style_instructions() if hasattr(self, 'style_plugin') and self.style_plugin is not None else None
                self._overlay_cache[cache_key] = make_overlay_parts(brand_instr, style_instr)

            brand_prefix, style_suffix = self._overlay_cache[cache_key]
            return enhance_prompt_with_parts(system_prompt, brand_prefix, style_suffix)
        except Exception:
            logger.warning("Failed to enhance prompt with brand personality and style", exc_info=True)
            return system_prompt

    def _build_messages_from_history(self, system_prompt: str, history: List[Message]) -> List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam]]:
        messages: List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam]] = [
            ChatCompletionSystemMessageParam(role="system", content=str(system_prompt))
        ]
        for msg in history:
            if msg.role == "user":
                messages.append(ChatCompletionUserMessageParam(role="user", content=msg.content))
            elif msg.role == "assistant":
                cleaned_content = clean_assistant_response(msg.content) if msg.content else ""
                if cleaned_content:
                    messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=cleaned_content))
        return messages

    @wrap_content_safety
    async def __call__(
        self,
        chat_history: List[Message],
        current_order: Optional[Dict] = None,
        delay: float = 0.05,
        model_deployment: Optional[str] = None,     
        conversation_style: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Handle a conversation turn."""
        current_order = current_order if current_order is not None else {"items": []}
        effective_style = conversation_style or self.conversation_style

        # Resolve current brand name (if brand plugin is available)
        if hasattr(self, 'brand_plugin'):
            brand_name = self.brand_plugin.get_current_brand()
        else:
            brand_name = None
        current_brand_name = brand_name or (self.brand_plugin.current_brand if hasattr(self, 'brand_plugin') else None)

        # Prepare context arguments
        context_args = KernelArguments()
        context_args["conversation_style"] = effective_style
        # Use the last N messages to keep prompts focused
        reduced_history = (chat_history or [])[-12:]
        context_args["chat_history"] = reduced_history
        context_args["brand_name"] = current_brand_name
        try:
            # Handle different flow types based on class name
            if self.__class__.__name__ == "OrderAssistantFlowSK":
                context_args["current_order"] = current_order
                plugin_name = "order_assistant"
                function_name = "chat"
            else:
                plugin_name = self.PLUGIN_NAME
                function_name = "chat"

            brand_personality = ""
            if hasattr(self, 'brand_plugin') and self.brand_plugin.current_brand:
                brand_personality = self.brand_plugin.get_brand_instructions()

            context_args["brand_personality"] = brand_personality
            # Get system prompt from template and enhance with brand personality
            system_prompt = await self.kernel.invoke(
                plugin_name=plugin_name,
                function_name=function_name,
                arguments=context_args
            )
            enhanced_prompt = await self.enhance_prompt_with_brand_and_style(str(system_prompt))

            # Build messages from recent history
            recent_history = (reduced_history or chat_history)[-8:]
            messages = self._build_messages_from_history(enhanced_prompt, recent_history)

            # completion parameters (centralized defaults + specific overrides)
            params = build_chat_params({"max_tokens": self.MAX_TOKENS})
            completion = self.client.chat.completions.create(
                model=model_deployment or self.DEPLOYMENT_NAME,
                messages=messages,
                **params,
            )

            # Process and yield tokens
            async for token in process_chat_stream(completion, delay):
                yield token

        except Exception as e:
            error_msg = f"Error in conversation: {str(e)}"
            logger.error(error_msg)
            yield "I apologize, but I encountered an error."


class PreamblePlugin(ConversationPlugin):
    """Plugin for preamble/greeting conversation functions."""
    
    def __init__(self, kernel: Kernel):
        super().__init__(kernel)
    
    @kernel_function(description="Prepares preamble context for prompt template", name="chat")
    async def prepare_preamble_context(self, chat_history: list[Message], BRAND_NAME: Optional[str] = None) -> str:
        """Formats preamble context for prompt template."""
        # Clean and format chat history
        clean_history = []
        for msg in chat_history:
            content = msg.content
            if msg.role == "assistant":
                
                content = clean_assistant_response(content)
            clean_history.append(f"{msg.role}: {content}")
        
        # Get recent conversation history 
        chat_str = "\n".join(clean_history[-4:])  # Shorter for preamble
        
        # Get brand name fallback
        brand_name = BRAND_NAME 
        brand_personality = ""
        
        template_context = f"""
Brand Name: {brand_name}
Brand Personality: {brand_personality}
Chat History: {chat_str}
        """.strip()
        
        return template_context
    
class PreambleFlowSK(ConversationFlowSK):
    """Initial greeting."""
    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/preamble_SK.prompty")
    PLUGIN_NAME = "preamble"

    def __init__(self, ENDPOINT: str, API_KEY: str, DEPLOYMENT_NAME: str, BRAND_NAME: Optional[str] = None, CONVERSATION_STYLE: Optional[str] = None):
        super().__init__(ENDPOINT, API_KEY, DEPLOYMENT_NAME, BRAND_NAME, CONVERSATION_STYLE)

        # Add PreamblePlugin for context formatting
        self.preamble_plugin = PreamblePlugin(self.kernel)
        self.kernel.add_plugin(self.preamble_plugin, "preamble")
        
        self._setup_prompt_function()
        logger.info("Initialized PreambleFlowSK for greetings only")


class OrderAssistantFlowSK(ConversationFlowSK):
    """Main order taking and menu assistance flow."""
    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/assistant_SK.prompty")
    PLUGIN_NAME = "order_assistant"

    def __init__(self, ENDPOINT: str, API_KEY: str, DEPLOYMENT_NAME: str, BRAND_NAME: Optional[str] = None, CONVERSATION_STYLE: Optional[str] = None):
        super().__init__(ENDPOINT, API_KEY, DEPLOYMENT_NAME, BRAND_NAME, CONVERSATION_STYLE)
        # Add OrderPlugin for menu and order context
        self.order_plugin = OrderPlugin(self.kernel)
        self.kernel.add_plugin(self.order_plugin, "order_assistant")
        # Verify menu is loaded
        if not hasattr(self.order_plugin, 'menu') or not self.order_plugin.menu:
            raise ValueError("Menu not properly loaded in OrderPlugin")
        logger.info("Initialized OrderAssistantFlowSK with menu context")
        
    async def invoke_semantic_function(self, function_name: str, arguments: KernelArguments) -> str:
        """Override to properly format the response."""
        # Get the raw response from the kernel
        response = await self.kernel.invoke(
            plugin_name=self.PLUGIN_NAME,
            function_name=function_name,
            arguments=arguments
        )

        # Clean the response to remove any embedded context
        response_text = str(response)
        return clean_assistant_response(response_text)


class SummaryFlowSK(ConversationFlowSK):
    """Order summary and confirmation flow."""
    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/summary_SK.prompty")
    PLUGIN_NAME = "summary"

    def __init__(self, ENDPOINT: str, API_KEY: str, DEPLOYMENT_NAME: str, BRAND_NAME: Optional[str] = None, CONVERSATION_STYLE: Optional[str] = None):
        super().__init__(ENDPOINT, API_KEY, DEPLOYMENT_NAME, BRAND_NAME, CONVERSATION_STYLE)