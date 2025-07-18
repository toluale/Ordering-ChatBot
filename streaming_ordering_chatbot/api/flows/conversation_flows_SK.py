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


def clean_assistant_response(content: str) -> str:
    """Remove context markers and sections from assistant responses.
    """
    if not content:
        return ""
        
    # List of markers that indicate context sections to remove
    context_markers = [
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
        "[END CONTEXT]",
        "MENU", 
        "CURRENT ORDER", 
        "CHAT HISTORY", 
        "CONVERSATION HISTORY"
    ]
    
    # Get the content before any context marker
    cleaned = content
    for marker in context_markers:
        if marker in cleaned:
            # Keep only the first part (before the marker)
            parts = cleaned.split(marker)
            cleaned = parts[0].strip()
    
    return cleaned


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
        """Load menu using MenuManager for the current brand."""
        try:
            # Get current brand from MenuManager (required to be set)
            current_brand = self.menu_manager.require_current_brand()
            
            # Generate menu description from configuration
            menu_config = self.menu_manager.require_current_menu_config()
            
            return self._format_menu_from_config(menu_config)
            
        except Exception as e:
            logger.error(f"Failed to load menu using MenuManager: {e}")
            raise RuntimeError(
                f"Menu loading failed: {e}. "
                "Please ensure RESTAURANT_BRAND environment variable is set "
                "and corresponding menu file exists in the data directory."
            ) from e
    
    def _format_menu_from_config(self, menu_config: Dict[str, Any]) -> str:
        """Format menu configuration into a readable string."""
        try:
            brand_info = menu_config.get("brand_info", {})
            menu_items = menu_config.get("menu_items", {})
            toppings = menu_config.get("toppings", {})
            
            menu_text = f"# {brand_info.get('name', 'Restaurant')} Menu\n\n"
            
            # Group items by category
            categories = {}
            for item_name, item_config in menu_items.items():
                category = item_config.get("category", "other")
                if category not in categories:
                    categories[category] = []
                categories[category].append((item_name, item_config))
            
            # Format each category
            for category, items in categories.items():
                menu_text += f"## {category.title()}s\n"
                for item_name, item_config in items:
                    name_variations = item_config.get("name_variations", [item_name])
                    menu_text += f"- {name_variations[0].title()}"
                    if len(name_variations) > 1:
                        menu_text += f" (also: {', '.join(name_variations[1:])})"
                    menu_text += "\n"
                menu_text += "\n"
            
            # Add toppings section
            if toppings:
                menu_text += "## Available Toppings\n"
                for topping_code, topping_info in toppings.items():
                    menu_text += f"- {topping_info.get('name', topping_code)}: {topping_info.get('description', '')}\n"
                menu_text += "\n"
            
            return menu_text
            
        except Exception as e:
            logger.error(f"Error formatting menu from config: {e}")
            raise RuntimeError(f"Failed to format menu from configuration: {e}") from e
        
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
        chat_str = "\n".join(clean_history[-6:])  # Keep last 6 messages for immediate context
        
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
        self.client = AzureOpenAI(
            api_key=self.API_KEY,
            api_version="2024-12-01-preview",
            azure_endpoint=self.ENDPOINT
        )
        
        # Set up plugins
        self.conversation_plugin = ConversationPlugin(self.kernel)
        self.kernel.add_plugin(self.conversation_plugin, "conversation")
        
        # Add brand personality plugin
        from .brand_personality import BrandPersonalityPlugin
        self.brand_plugin = BrandPersonalityPlugin(self.kernel, BRAND_NAME)
        self.kernel.add_plugin(self.brand_plugin, "brand")
        
        # Add conversation style plugin only if a specific style is provided
        # If no style is provided, will default to brand's original style
        self.style_plugin = None
        if CONVERSATION_STYLE and CONVERSATION_STYLE.lower() not in ["default", "none", ""]:
            try:
                from .conversation_style import ConversationStylePlugin, ConversationStyle
                style_enum = ConversationStyle(CONVERSATION_STYLE.lower())
                self.style_plugin = ConversationStylePlugin(self.kernel, style_enum)
                self.kernel.add_plugin(self.style_plugin, "style")
                logger.info(f"Added conversation style plugin: {CONVERSATION_STYLE}")
            except (ValueError, ImportError) as e:
                logger.warning(f"Failed to load conversation style '{CONVERSATION_STYLE}': {e}. Using brand's original style.")
                self.style_plugin = None
        else:
            logger.info("No specific conversation style provided. Using brand's original style.")
        
        # Load prompt template
        if self.PROMPT_PATH:
            self._setup_prompt_function()

    def _setup_prompt_function(self) -> None:
        """Set up the prompt function with brand personality and conversation style integration."""
        try:
            if not self.PROMPT_PATH:
                raise ValueError("PROMPT_PATH must be set in the derived class")
                
            with open(self.PROMPT_PATH, "r", encoding="utf-8") as f:
                base_template = f.read()
            
            # Get brand instructions and optionally enhance with conversation style
            if hasattr(self, 'brand_plugin'):
                brand_instructions = self.brand_plugin.get_brand_instructions()
                
                # Apply conversation style to brand instructions only if style plugin exists
                # Otherwise, use the brand's original style without modification
                if hasattr(self, 'style_plugin') and self.style_plugin is not None:
                    enhanced_instructions = self.style_plugin.enhance_brand_with_style(brand_instructions)
                    prompt_template = f"{enhanced_instructions}\n\n{base_template}"
                else:
                    # Use original brand instructions without style modification
                    prompt_template = f"{brand_instructions}\n\n{base_template}"
            else:
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
        """Enhance system prompt with both brand personality and conversation style."""
        try:
            enhanced_prompt = system_prompt
            
            # First apply brand enhancement if available
            if hasattr(self, 'brand_plugin'):
                args = KernelArguments()
                args["system_prompt"] = enhanced_prompt
                result = await self.kernel.invoke(
                    plugin_name="brand",
                    function_name="enhance_system_prompt",                
                    arguments=args
                )
                enhanced_prompt = str(result) if result is not None else enhanced_prompt
            
            # Then apply conversation style enhancement if available
            if hasattr(self, 'style_plugin') and self.style_plugin is not None:
                style_instructions = self.style_plugin.get_style_instructions()
                if style_instructions:
                    enhanced_prompt = f"{enhanced_prompt}\n\nCONVERSATION STYLE:\n{style_instructions}"
            
            return enhanced_prompt
            
        except Exception:
            logger.warning("Failed to enhance prompt with brand personality and style", exc_info=True)
            return system_prompt

    @staticmethod
    def fix_concatenated_words(text: str) -> str:
        """Post-process text to fix common word concatenation issues."""
        import re
        
        # Fix punctuation followed immediately by letters (Restaurant!We're -> Restaurant! We're)
        text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)
        
        # Fix comma/semicolon/colon followed immediately by letters (fresh,we're -> fresh, we're)
        text = re.sub(r'([,;:])([a-zA-Z])', r'\1 \2', text)
        
        # Fix lowercase letter followed immediately by uppercase (andComfort -> and Comfort)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Fix word boundaries with numbers (for4 -> for 4, 4people -> 4 people)
        text = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', text)
        text = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        
        # Fix specific markdown/formatting issues
        text = re.sub(r'(\*\*)([A-Z])', r'\1 \2', text)  # **Drinks: -> ** Drinks:
        text = re.sub(r'([!.?])(\*\*)', r'\1 \2', text)  # sharing!** -> sharing! **
        
        # Fix compound words that should have spaces (Lemon-Limesoda -> Lemon-Lime soda)
        text = re.sub(r'([a-z])([A-Z][a-z]+)([a-z])', lambda m: f"{m.group(1)} {m.group(2).lower()}{m.group(3)}", text)
        
        return text

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
                
                # More intelligent buffering - yield on proper word boundaries
                should_yield = False
                
                # Yield on sentence endings
                if token in '.!?\n':
                    should_yield = True
                # Yield on clause boundaries
                elif token in ',;:' and len(buffer) > 2:
                    should_yield = True
                # Yield when we hit a space after building up some content
                elif token == ' ' and len(buffer) > 3:
                    should_yield = True
                # Safety valve for very long buffers
                elif len(buffer) > 15:
                    should_yield = True
                if should_yield:
                    text = "".join(buffer)
                    if text and text !=' ':
                        # Apply post-processing to fix concatenation issues
                        fixed_text = ConversationFlowSK.fix_concatenated_words(text)
                        yield fixed_text
                    buffer = []
                    
            # Yield remaining content with post-processing
            if buffer:
                final_text = "".join(buffer)
                if final_text and final_text != ' ':
                    fixed_final_text = ConversationFlowSK.fix_concatenated_words(final_text)
                    yield fixed_final_text
                    
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
        current_brand_name = brand_name or (self.brand_plugin.current_brand if hasattr(self, 'brand_plugin') else None)
        try:
            # Prepare context arguments
            context_args = KernelArguments()
            
            # Handle different flow types based on class name
            if self.__class__.__name__ == "OrderAssistantFlowSK":
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
            enhanced_prompt = await self.enhance_prompt_with_brand_and_style(str(system_prompt))
            
            # Create messages list with system message and recent context
            messages: List[Union[ChatCompletionSystemMessageParam, ChatCompletionUserMessageParam, ChatCompletionAssistantMessageParam]] = [
                ChatCompletionSystemMessageParam(role="system", content=str(enhanced_prompt))
            ]

            # Get the last 8 messages from chat history
            recent_history = chat_history[-8:]
            
            # Add cleaned conversation history
            for msg in recent_history:
                if msg.role == "user":
                    messages.append(ChatCompletionUserMessageParam(role="user", content=msg.content))
                elif msg.role == "assistant":
                    cleaned_content = clean_assistant_response(msg.content) if msg.content else ""
                    if cleaned_content:
                        messages.append(ChatCompletionAssistantMessageParam(role="assistant", content=cleaned_content))

            # completion parameters
            completion = self.client.chat.completions.create(
                model=model_deployment or self.DEPLOYMENT_NAME,
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
                # Use the shared cleaning utility function
                content = clean_assistant_response(content)
            clean_history.append(f"{msg.role}: {content}")
        
        # Get recent conversation history (last 3-4 messages for preamble)
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
        for marker in ["Chat History:", "Current Order:", "Menu:", "Previous conversation:", "Available menu:", "Current order status:"]:
            if marker in response_text:
                response_text = response_text.split(marker)[0].strip()
            
        return response_text


class SummaryFlowSK(ConversationFlowSK):
    """Order summary and confirmation flow."""
    PROMPT_PATH = Path(__file__).parent.joinpath("prompts/summary_SK.prompty")
    PLUGIN_NAME = "summary"

    def __init__(self, ENDPOINT: str, API_KEY: str, DEPLOYMENT_NAME: str, BRAND_NAME: Optional[str] = None, CONVERSATION_STYLE: Optional[str] = None):
        super().__init__(ENDPOINT, API_KEY, DEPLOYMENT_NAME, BRAND_NAME, CONVERSATION_STYLE)