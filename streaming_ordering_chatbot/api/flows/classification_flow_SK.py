from pathlib import Path
from streaming_ordering_chatbot.api.flows.schemas_generalized import LLMOrder, set_brand_context 
from streaming_ordering_chatbot.api.models import Message

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments


class OrderIntentPlugin:
    """Plugin for order intent classification."""
    
    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self.prompt_path = Path(__file__).parent.joinpath("prompts/order_intent_SK.prompty")
        
        # Load prompt template
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        
        # Register the prompt template as a kernel function during plugin initialization
        self.prompt_config = PromptTemplateConfig(
            template=self.prompt_template,
            template_format="semantic-kernel"
        )
        
        # Register the prompt template as a function in the kernel
        self.kernel.add_function(
            plugin_name="order_intent_prompts",
            function_name="classify_prompt",
            prompt_template_config=self.prompt_config
        )
    
    @kernel_function(name="classify_intent", description="Classifies user intent for order modification.")
    async def classify_intent(self, user_message: str, current_order: dict) -> str:
        """Classifies whether user message is about ordering or general conversation."""
        # Now use the registered prompt function
        response = await self.kernel.invoke(
            plugin_name="order_intent_prompts",
            function_name="classify_prompt",
            arguments=KernelArguments(
                user_message=user_message,
                current_order=current_order
            )
        )
        
        msg = str(response)
        return "order" if "<yes>" in msg.lower() else "conversation"


class OrderIntentFlowSK:
    """Order intent classification flow using Semantic Kernel."""
    
    def __init__(self, ENDPOINT: str, API_KEY: str, DEPLOYMENT_NAME: str, BRAND_NAME: str = "default"):
        """Initialize the order intent classification flow."""
        self.ENDPOINT = ENDPOINT
        self.API_KEY = API_KEY
        self.DEPLOYMENT_NAME = DEPLOYMENT_NAME
        self.BRAND_NAME = BRAND_NAME

        if BRAND_NAME:
            try:
                set_brand_context(BRAND_NAME)
            except Exception as e:
                # brand_config not needed for intent classification
                pass
        # Initialize Semantic Kernel
        self.kernel = Kernel()
        
        # Set up Azure Chat Service
        self.azure_chat = AzureChatCompletion(
            deployment_name=self.DEPLOYMENT_NAME,
            endpoint=self.ENDPOINT,
            api_key=self.API_KEY
        )
        self.kernel.add_service(self.azure_chat)
        
        # Register the classification plugin
        # This automatically registers all @kernel_function decorated methods
        self.order_intent_plugin = OrderIntentPlugin(self.kernel)
        self.kernel.add_plugin(self.order_intent_plugin, "order_intent")

    async def __call__(self, chat_history: list[Message], current_order: dict) -> str:
        """Execute order intent classification."""
        # Validate and prepare order data
        try:
            current_order = LLMOrder.model_validate(current_order).model_dump()
        except Exception as e:
            if "No brand context set" in str(e):
                pass
            else:
                raise 
        user_message = chat_history[-1].content
        
        # Now this properly calls the registered plugin function
        result = await self.kernel.invoke(
            plugin_name="order_intent",
            function_name="classify_intent",
            arguments=KernelArguments(
                user_message=user_message,
                current_order=current_order
            )
        )
        
        return str(result)