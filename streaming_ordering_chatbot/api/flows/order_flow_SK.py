import asyncio
import json
import logging
import os
from pathlib import Path
from typing import AsyncGenerator, Union, Optional, Dict, List, Any
from copy import copy

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments
from openai import AzureOpenAI
from openai.types.chat import ChatCompletionUserMessageParam

from streaming_ordering_chatbot.api.flows.schemas_generalized import (LLMBurgerItem, LLMDrinkItem, LLMFriesItem, LLMOrder, LLMGenericItem)
from streaming_ordering_chatbot.api.models import Message
from streaming_ordering_chatbot.api.flows.menu_manager import get_menu_manager
from streaming_ordering_chatbot.api.flows.schemas_generalized import set_brand_context

# Set up logging
handler = logging.FileHandler("streaming_ordering_chatbot.order_flow_SK.log")
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)

logger = logging.getLogger(__name__)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def validate_brand_name(brand_name: Optional[str], context: str) -> str:
    """Validate brand name and raise appropriate error if missing."""
    if not brand_name:
        raise ValueError(f"Brand name is required for {context}")
    return brand_name


def safe_json_parse(data: Union[str, Any]) -> Any:
    """Safely parse JSON data, returning as-is if already parsed."""
    return json.loads(data) if isinstance(data, str) else data


def messages_to_dicts(messages: List[Message]) -> List[Dict[str, str]]:
    """Convert Message objects to dictionaries."""
    result = []
    for msg in messages:
        if hasattr(msg, 'model_dump'):
            result.append(msg.model_dump())
        else:
            result.append({"role": msg.role, "content": msg.content})
    return result


def create_azure_openai_client(api_key: str, endpoint: str) -> AzureOpenAI:
    """Create and return AzureOpenAI client with standard configuration."""
    return AzureOpenAI(
        api_key=api_key,
        api_version="2024-12-01-preview",
        azure_endpoint=endpoint
    )


class OrderValidationPlugin:
    """Plugin for order validation and processing."""
    
    def __init__(self, kernel: Kernel, brand_name: Optional[str] = None):
        self.kernel = kernel
        self.brand_name = validate_brand_name(brand_name, "OrderValidationPlugin")
        
        try:
            menu_manager = get_menu_manager()
            self.menu = menu_manager.get_menu_text_format(self.brand_name)
        except Exception as e:
            logger.error(f"Could not load menu for brand {self.brand_name}: {e}")
            raise ValueError(f"Failed to load menu configuration for brand: {self.brand_name}") from e
        
        # Load tools
        self.tools_path = Path(__file__).parent.joinpath("prompts/tools.json")
        if self.tools_path.exists():
            with open(self.tools_path, "r") as f:
                self.tools = json.load(f)
        else:
            self.tools = {}
    
    @kernel_function(name="validate_item", description="Validates a menu item against the order schema")
    async def validate_item(self, item: str) -> str:
        """Validates an item and returns validation result."""
        try:
            item_dict = safe_json_parse(item)
            logger.info("Validating item: %s", item_dict)
            
            # Remove itemId for validation as it's not part of the LLM item schema
            if "itemId" in item_dict:
                item_dict.pop("itemId")
            # Attempt validation using LLMOrder schema
            try:
                items = LLMOrder.model_validate({"items": [item_dict]}).items
                if not items:
                    logger.warning("Invalid item: %s", item_dict)
                    return json.dumps({"valid": False, "error": "Empty items list"})
                
                validated_item = items[0].to_order_item()
                if validated_item:
                    logger.info("Item validated successfully: %s", validated_item.model_dump())
                    
                    return json.dumps({
                        "valid": True, 
                        "item": validated_item.model_dump(),
                        "original_item": items[0].model_dump(),  # Preserve original LLM item data
                        "item_type": type(items[0]).__name__
                    })
                else:
                    logger.warning("Item validation returned None: %s", item_dict)
                    return json.dumps({"valid": False, "error": "Validation returned None"})
           
            except Exception as validation_error:
                logger.error("LLMOrder validation failed for item %s: %s", item_dict, validation_error)
                return json.dumps({"valid": False, "error": str(validation_error)})   
        
        except Exception as e:
            logger.error("Error validating item: %s", e)
            return json.dumps({"valid": False, "error": str(e)})
    
    @kernel_function(name="get_menu", description="Returns the restaurant menu")
    async def get_menu(self) -> str:
        """Returns the menu content."""
        return self.menu
    
    @kernel_function(name="get_tools", description="Returns the available tools for order processing")
    async def get_tools(self) -> str:
        """Returns the tools configuration."""
        return json.dumps(self.tools)


class OrderProcessingPlugin:
    """Plugin for order processing and generation."""
    
    def __init__(self, kernel: Kernel, brand_name: Optional[str] = None):
        self.kernel = kernel
        self.brand_name = validate_brand_name(brand_name, "OrderProcessingPlugin")
        self.prompt_path = Path(__file__).parent.joinpath("prompts/order_SK.prompty")
        
        if self.prompt_path.exists():
            with open(self.prompt_path, "r", encoding="utf-8") as f:
                self.prompt_template = f.read()
            
            # Register the prompt template as a kernel function
            self.prompt_config = PromptTemplateConfig(
                name="process_order",
                description="Process order using SK template",
                template=self.prompt_template
            )
            
            self.kernel.add_function(
                plugin_name="order_processing_prompts",
                function_name="process_order",
                prompt_template_config=self.prompt_config
            )
        else:
            logger.warning("Order processing prompt template not found")
            self.prompt_template = None
            self.prompt_config = None
    
    @kernel_function(name="format_chat_history", description="Formats chat history for order processing")
    async def format_chat_history(self, chat_history: str) -> str:
        """Formats chat history for the order prompt."""
        try:
            history_list = safe_json_parse(chat_history)
            formatted_history = []
            
            for msg in history_list:
                if isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    if role == 'user':
                        formatted_history.append(f"customer: {content}")
                    elif role == 'assistant':
                        formatted_history.append(f"OrderBot: {content}")
                elif hasattr(msg, 'role') and hasattr(msg, 'content'):
                    if msg.role == 'user':
                        formatted_history.append(f"customer: {msg.content}")
                    elif msg.role == 'assistant':
                        formatted_history.append(f"OrderBot: {msg.content}")
            
            return "\n".join(formatted_history)
        except Exception as e:
            logger.error("Error formatting chat history: %s", e)
            return ""


class OrderFlowSK:
    """implementation of order flow with brand context support."""
    
    def __init__(self, endpoint: str, api_key: str, deployment_name: str, brand_name: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.brand_name = validate_brand_name(brand_name, "OrderFlowSK")
        
        # Initialize brand context
        try:
            set_brand_context(self.brand_name)
            logger.info(f"Set brand context to: {self.brand_name}")
        except Exception as e:
            logger.error(f"Could not set brand context to {self.brand_name}: {e}")
            raise ValueError(f"Failed to set brand context: {self.brand_name}") from e
        
        self.kernel = Kernel()
        
        self.azure_chat = AzureChatCompletion(
            deployment_name=self.deployment_name,
            endpoint=self.endpoint,
            api_key=self.api_key
        )
        self.kernel.add_service(self.azure_chat)
        
        self.client = create_azure_openai_client(self.api_key, self.endpoint)
        
        # Register plugins
        self.validation_plugin = OrderValidationPlugin(self.kernel, self.brand_name)
        self.kernel.add_plugin(self.validation_plugin, "validation")
        
        self.processing_plugin = OrderProcessingPlugin(self.kernel, self.brand_name)
        self.kernel.add_plugin(self.processing_plugin, "processing")
        
        # Load menu and tools
        self.menu = self.validation_plugin.menu
        self.tools = self.validation_plugin.tools
        
        logger.info(f"OrderFlowSK initialized successfully for brand: {self.brand_name}")
    
    async def validate_item(self, item: dict) -> Union[None, LLMBurgerItem, LLMDrinkItem, LLMFriesItem]:
        """Validates an item using the validation plugin."""
        try:
            result = await self.kernel.invoke(
                plugin_name="validation",
                function_name="validate_item",
                arguments=KernelArguments(item=json.dumps(item))
            )
            
            validation_result = json.loads(str(result))
            
            if validation_result.get("valid", False):
                original_item_data = validation_result.get("original_item", {})
                item_type = validation_result.get("item_type", "")
                
                # Use the original LLM item data for reconstruction
                if item_type == "LLMBurgerItem":
                    return LLMBurgerItem.model_validate(original_item_data)
                elif item_type == "LLMDrinkItem":
                    return LLMDrinkItem.model_validate(original_item_data)
                elif item_type == "LLMFriesItem":
                    return LLMFriesItem.model_validate(original_item_data)
                else:
                    # Try to determine type from the original item data
                    if "toppings" in original_item_data or "bun" in original_item_data:
                        return LLMBurgerItem.model_validate(original_item_data)
                    elif "size" in original_item_data and "name" in original_item_data:
                        if "fries" in original_item_data["name"].lower():
                            return LLMFriesItem.model_validate(original_item_data)
                        else:
                            return LLMDrinkItem.model_validate(original_item_data)
            
            return None
            
        except Exception as e:
            logger.error("Error validating item: %s", e)
            return None
    
    async def process_order_with_sk(self, chat_history: List[Message], current_order: dict) -> str:
        """Process order"""
        try:
            # Convert Message objects to dictionaries
            history_dicts = messages_to_dicts(chat_history)
            
            # Format inputs
            formatted_history = await self.processing_plugin.format_chat_history(json.dumps(history_dicts))
            
            # Use the registered prompt function 
            if hasattr(self.processing_plugin, 'prompt_template'):
                result = await self.kernel.invoke(
                    plugin_name="order_processing_prompts",
                    function_name="process_order",
                    arguments=KernelArguments(
                        brand_name=self.brand_name,
                        chat_history=formatted_history,
                        current_order=json.dumps(current_order),
                        menu=self.menu
                    )
                )
                return str(result)
            else:
                return await self._create_order_prompt(chat_history, current_order)
            
        except Exception as e:
            logger.error("Error processing order: %s", e)
            return json.dumps({"error": str(e)})
    
    async def _create_order_prompt(
        self, 
        chat_history: List[Message], 
        current_order: dict
    ) -> str:
        """Create the order processing prompt using the same logic as the template.
        """
        # Convert Message objects to dictionaries for processing
        history_dicts = messages_to_dicts(chat_history)
        
        # Format chat history using the plugin
        formatted_history = await self.processing_plugin.format_chat_history(json.dumps(history_dicts))
        
        # Use the same prompt template structure from order_SK.prompty
        prompt_content = f"""{formatted_history}

Current Order:
{json.dumps(current_order)}

Please process this order request and provide a structured response with the updated order items."""
        
        return prompt_content

    async def _format_messages_for_streaming(
        self, 
        chat_history: List[Message], 
        current_order: dict
    ) -> List[ChatCompletionUserMessageParam]:
        """Format messages for Azure OpenAI streaming call.
        Args:
            chat_history: List of chat messages
            current_order: Current order state 
        Returns:
            Formatted messages list
        """
        prompt_content = await self._create_order_prompt(chat_history, current_order)
        return [ChatCompletionUserMessageParam(role="user", content=prompt_content)]

    async def stream_order_response(
        self,
        chat_history: List[Message],
        current_order: dict,
        delay: float = 0.05
    ) -> AsyncGenerator[str, None]:
        """Stream order response using Azure OpenAI with function calling.
        Args:
            chat_history: List of chat messages
            current_order: Current order state
            delay: Streaming delay
        Yields:
            Streaming order response
        """
        try:
            # Validate current order
            current_order = LLMOrder.model_validate(current_order).model_dump()
            
            # Format messages using consistent prompt logic
            messages = await self._format_messages_for_streaming(chat_history, current_order)
            
            # Stream response with function calling for structured output
            response = self.client.chat.completions.create(
                model=self.deployment_name,
                messages=messages,
                tools=self.tools.get("tools", []),
                tool_choice=self.tools.get("tool_choice", "auto"),
                stream=True,
                max_tokens=1000
            )
            
            # Process streaming response for structured order items
            async for chunk in self._process_streaming_chunks(response, delay):
                yield chunk
            
        except Exception as e:
            logger.error("Error in stream_order_response: %s", e)
            yield json.dumps({"error": str(e)}) + "\n"

    async def _process_streaming_chunks(
        self, 
        response, 
        delay: float
    ) -> AsyncGenerator[str, None]:
        """Process streaming chunks and validate order items.
        """
        json_string = ""
        current_order_items = {"items": []}
        prev_item_id = -1
        first_item = True
        
        for chunk in response:
            await asyncio.sleep(delay)
            
            if (chunk is not None and 
                hasattr(chunk, "choices") and 
                len(chunk.choices) > 0 and
                chunk.choices[0].delta.tool_calls):
                
                try:
                    tool_call = chunk.choices[0].delta.tool_calls[0]
                    if (hasattr(tool_call, 'function') and 
                        tool_call.function and 
                        hasattr(tool_call.function, 'arguments') and
                        tool_call.function.arguments):
                        
                        arg = tool_call.function.arguments
                        logger.debug("Processing function arguments")  # Avoid logging potentially sensitive tokens
                        
                        if isinstance(arg, str):
                            json_string += arg
                            
                            # Process when we have a complete item
                            if "}" in arg:
                                parse_string = json_string[:json_string.rfind("}") + 1]
                                left_brace = parse_string.count("{")
                                right_brace = parse_string.count("}")
                                
                                if right_brace > 0 and (left_brace - right_brace) == 1:
                                    try:
                                        current_order_items = json.loads(parse_string + "]}")
                                        
                                        # Process new items
                                        if (items := current_order_items.get("items")) and items:
                                            latest_item = items[-1]
                                            current_item_id = latest_item.get("itemId")
                                            
                                            if current_item_id != prev_item_id:
                                                prev_item_id = current_item_id
                                                
                                                # Validate and stream item
                                                logger.info(f"Attempting to validate item: {latest_item}")
                                                validated_item = await self.validate_item(latest_item)
                                                                                            
                                                if validated_item is not None:
                                                    ser_item = validated_item.model_dump()
                                                    
                                                    if first_item:
                                                        first_item = False
                                                        yield '{"order": [' + json.dumps(ser_item) + "\n"
                                                    else:
                                                        yield "," + json.dumps(ser_item) + "\n"
                                                    
                                                    logger.info("Validated and streamed item: %s", ser_item)
                                                else:
                                                    logger.warning(f"Item validation failed for: {latest_item}")
                                    except json.JSONDecodeError as e:
                                        logger.warning("JSON decode error: %s", e)
                                        continue
                
                except (AttributeError, IndexError) as e:
                    logger.warning("Error processing chunk: %s", e)
                    continue
        
        # Close JSON structure and return final order state
        if not first_item:
            yield "]}\n"
        
        # Always return the final LLMOrder structure
        try:
            final_order = LLMOrder.model_validate(current_order_items)
            yield json.dumps({"LLMOrder": final_order.model_dump()}) + "\n"
        except Exception as e:
            logger.error(f"Error creating final order: {e}")
            yield json.dumps({"LLMOrder": {"items": []}}) + "\n"
    
    async def __call__(
        self,
        chat_history: List[Message],
        current_order: dict,
        delay: float = 0.05,
        model_deployment: Optional[str] = None,
        use_streaming: bool = True
    ) -> AsyncGenerator[str, None]:
        original_deployment = None
        try:
            
            if model_deployment:
                original_deployment = self.deployment_name
                self.deployment_name = model_deployment
                
                # Update client
                self.client = create_azure_openai_client(self.api_key, self.endpoint)
            
            if use_streaming:
                # Use streaming response
                async for chunk in self.stream_order_response(chat_history, current_order, delay):
                    yield chunk
            else:
                result = await self.process_order_with_sk(chat_history, current_order)
                yield result
                
        except Exception as e:
            logger.error("Error in OrderFlowSK.__call__: %s", e)
            yield json.dumps({"error": str(e)}) + "\n"
        
        finally:
            if original_deployment is not None:
                self.deployment_name = original_deployment
