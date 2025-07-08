import os
import asyncio
import logging
import json
from pathlib import Path
from dotenv import load_dotenv
from typing import Optional, AsyncGenerator, Dict, List, Union
from openai import AzureOpenAI
from openai.types.chat import (ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam)
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

    def _clean_assistant_response(self, content: str) -> str:
        """Remove context markers from responses while preserving order information."""
        if not content:
            return ""
            
        # List of markers to remove
        markers = [
            "Previous conversation:",
            "Current Order:",
            "Available menu:",
            "Current order status:",
            "Menu:",
            "Chat History:",
            "Instructions:",
            "Reference Information"
        ]
        
        # Get the content before any system markers
        cleaned = content
        for marker in markers:
            if marker in cleaned:
                parts = cleaned.split(marker)
                # Keep only the first part (before the marker)
                cleaned = parts[0].strip()
        
        return cleaned
        
    @kernel_function(description="Prepares order context for prompt template", name="chat")
    async def prepare_order_context(self, chat_history: list[Message], current_order: dict, brand_name: Optional[str] = None) -> str:
        """Formats order context for prompt template."""
        # Clean and format chat history
        clean_history = []
        for msg in chat_history:
            content = msg.content
            if msg.role == "assistant":
                content = self._clean_assistant_response(content)
            clean_history.append(f"{msg.role}: {content}")
        
        # Get recent conversation history
        chat_str = "\n".join(clean_history[-6:])  # Keep last 6 messages for immediate context
        
        items = current_order.get("items", [])
        order_str = "Current items in order: " + ", ".join(str(item) for item in items) if items else "No items in order yet"
        
        # Get brand name from brand plugin if not provided
        if not brand_name and hasattr(self, 'kernel'):
            try:
                # Try to get current brand from brand plugin
                brand_result = await self.kernel.invoke(
                    plugin_name="brand",
                    function_name="get_current_brand"
                )
                brand_name = str(brand_result).split(":")[1].strip() if ":" in str(brand_result) else ""
            except:
                brand_name = "Contoso Burger"  # Fallback

        # brand_name = brand_name or "Contoso Burger"  # Final fallback

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
    MAX_TOKENS = 500  

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
            api_version="2024-12-01-preview",
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
                function_name="enhance_system_prompt",                
                arguments=args
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
            buffer = []

            for chunk in completion:
                if delay > 0:
                    await asyncio.sleep(delay)
                
                if (not chunk or not chunk.choices 
                    or not chunk.choices[0].delta 
                    or not hasattr(chunk.choices[0].delta, 'content')
                    or chunk.choices[0].delta.content is None):
                    continue
                
                token = chunk.choices[0].delta.content
                buffer.append(token)
                
                # Yield on natural breaks or when buffer gets too large
                if (token in ".!?,;:\n" or len(buffer) > 10):
                    text = "".join(buffer)
                    if text.strip(): 
                        yield text
                    buffer = []
            
            # Yield any remaining content in buffer
            if buffer:
                final_text = "".join(buffer)
                if final_text.strip():
                    yield final_text
                
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
        """Handle a conversation turn."""
        current_order = current_order if current_order is not None else {"items": []}
        if hasattr(self, 'brand_plugin'):
            brand_name = self.brand_plugin.get_current_brand()
        else:
            brand_name = None
        current_brand_name = (brand_name or (self.brand_plugin.current_brand if hasattr(self, 'brand_plugin') else None)) #or "Contoso Burger"
        try:
            # Prepare context arguments
            context_args = KernelArguments()
            
            # Handle OrderAssistantFlowSK 
            if isinstance(self, OrderAssistantFlowSK):
                context_args["current_order"] = current_order
                context_args["chat_history"] = chat_history
                context_args["brand_name"] = current_brand_name
                plugin_name = "order_assistant"
                function_name = "chat"
            else:
                context_args["chat_history"] = chat_history
                context_args["brand_name"] = current_brand_name
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
            enhanced_prompt = await self.enhance_prompt_with_brand(str(system_prompt))
            
            # Create messages list with system message and recent context
            messages: List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam]] = [
                ChatCompletionSystemMessageParam(role="system", content=str(enhanced_prompt))
            ]

            # Get the last 6 messages from chat history
            recent_history = chat_history[-6:]
            
            # Add cleaned conversation history
            for msg in recent_history:
                if msg.role == "user":
                    messages.append(ChatCompletionUserMessageParam(role="user", content=msg.content))
                elif msg.role == "assistant":
                    cleaned_content = self._clean_assistant_response(msg.content) if msg.content else ""
                    if cleaned_content:
                        messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=cleaned_content))

            # completion parameters
            completion = self.client.chat.completions.create(
                model=model_deployment or self.deployment_name,
                messages=messages,
                temperature=0.7,
                top_p=0.95,
                max_tokens=self.MAX_TOKENS,
                presence_penalty=0.6,  
                frequency_penalty=0.3,  
                stream=True
            )

            # Process and yield tokens
            async for token in self._process_stream(completion, delay):
                yield token

        except Exception as e:
            error_msg = f"Error in conversation: {str(e)}"
            logger.error(error_msg)
            yield "I apologize, but I encountered an error."

    def _clean_assistant_response(self, content: str) -> str:
        """Remove context markers and sections from assistant responses."""
        # List of markers that indicate context sections
        context_markers = [
            "MENU", "CURRENT ORDER", "CHAT HISTORY", "CONVERSATION HISTORY",
            "Current Order:", "Menu:", "Chat History:", "Available menu:",
            "Current order status:", "Previous conversation:"
        ]
        
        # Get the content before any context marker
        cleaned = content
        for marker in context_markers:
            if marker in cleaned:
                cleaned = cleaned.split(marker)[0].strip()
        
        return cleaned


class PreamblePlugin(ConversationPlugin):
    """Plugin for preamble/greeting conversation functions."""
    
    def __init__(self, kernel: Kernel):
        super().__init__(kernel)
    
    @kernel_function(description="Prepares preamble context for prompt template", name="chat")
    async def prepare_preamble_context(self, chat_history: list[Message], brand_name: Optional[str] = None) -> str:
        """Formats preamble context for prompt template."""
        # Clean and format chat history
        clean_history = []
        for msg in chat_history:
            content = msg.content
            if msg.role == "assistant":
                # Use the same cleaning method from OrderPlugin
                content = self._clean_assistant_response(content)
            clean_history.append(f"{msg.role}: {content}")
        
        # Get recent conversation history (last 3-4 messages for preamble)
        chat_str = "\n".join(clean_history[-4:])  # Shorter for preamble
        
        # Get brand name fallback
        brand_name = brand_name or "Contoso Burger"
        brand_personality = ""
        
        template_context = f"""
Brand Name: {brand_name}
Brand Personality: {brand_personality}
Chat History: {chat_str}
        """.strip()
        
        return template_context
    
    def _clean_assistant_response(self, content: str) -> str:
        """Remove context markers from responses."""
        if not content:
            return ""
            
        markers = [
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
            "[END CONTEXT]"
        ]
        
        cleaned = content
        for marker in markers:
            if marker in cleaned:
                parts = cleaned.split(marker)
                cleaned = parts[0].strip()
        
        return cleaned


class PreambleFlowSK(ConversationFlowSK):
    """Initial greeting."""
    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/preamble_SK.prompty")
    PLUGIN_NAME = "preamble"

    def __init__(self, endpoint: str, api_key: str, deployment_name: str, brand_name: Optional[str] = None):
        super().__init__(endpoint, api_key, deployment_name, brand_name)
        
        # Add PreamblePlugin for context formatting
        self.preamble_plugin = PreamblePlugin(self.kernel)
        self.kernel.add_plugin(self.preamble_plugin, "preamble")
        
        self._setup_prompt_function()
        logger.info("Initialized PreambleFlowSK for greetings only")


class OrderAssistantFlowSK(ConversationFlowSK):
    """Main order taking and menu assistance flow."""
    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/assistant_SK.prompty")
    PLUGIN_NAME = "order_assistant"

    def __init__(self, endpoint: str, api_key: str, deployment_name: str, brand_name: Optional[str] = None):
        super().__init__(endpoint, api_key, deployment_name, brand_name)
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
        for marker in ["Chat History:", "Current Order:", "Menu:", "Previous conversation:", "Available menu:", "Current order status:"]:
            if marker in response_text:
                response_text = response_text.split(marker)[0].strip()
            
        return response_text


class SummaryFlowSK(ConversationFlowSK):
    """Order summary and confirmation flow."""
    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/summary_SK.prompty")
    PLUGIN_NAME = "summary"

    def __init__(self, endpoint: str, api_key: str, deployment_name: str, brand_name: Optional[str] = None):
        super().__init__(endpoint, api_key, deployment_name, brand_name)
