from pathlib import Path
from streaming_ordering_chatbot.api.flows.schemas import LLMOrder
from streaming_ordering_chatbot.api.models import Message

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.prompt_template import PromptTemplateConfig
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments

# Python plugin for business logic
class OrderUtils:
    @kernel_function(name="is_order_empty", description="Checks if the current order is empty.")
    def is_order_empty(self, current_order: dict) -> bool:
        return not current_order.get("items")

class OrderIntentFlowSK:
    # Constants for the prompt function (LLM)
    PROMPT_FUNCTION_NAME = "order_intent"
    PROMPT_PLUGIN_NAME = "order_plugin"
    
    # Constants for the utility plugin
    UTILS_PLUGIN_NAME = "OrderUtils"
    UTILS_FUNCTION_NAME = "is_order_empty"

    def __init__(self, ENDPOINT: str, API_KEY: str, DEPLOYMENT_NAME: str):
        """Classifies user intent for order modification using Semantic Kernel and Azure AI Foundry."""
        self.endpoint = ENDPOINT
        self.api_key = API_KEY
        self.deployment_name = DEPLOYMENT_NAME
        self.prompt_path = Path(__file__).parent.joinpath("prompts/order_intent_SK.prompty")
        self.kernel = Kernel()
        
        # Set up Azure Chat Service
        self.azure_chat = AzureChatCompletion(
            deployment_name=self.deployment_name,
            endpoint=self.endpoint,
            api_key=self.api_key
        )
        self.kernel.add_service(self.azure_chat)
        
        # Register Python utility plugin first
        self.kernel.add_plugin(OrderUtils(), self.UTILS_PLUGIN_NAME)
        
        # Load and register prompt template as a function
        with open(self.prompt_path, "r", encoding="utf-8") as f:
            self.prompt_template = f.read()
        self.prompt_config = PromptTemplateConfig(
            template=self.prompt_template,
            template_format="semantic-kernel"  
        )
        self.kernel.add_function(
            plugin_name=self.PROMPT_PLUGIN_NAME,
            function_name=self.PROMPT_FUNCTION_NAME,
            prompt_template_config=self.prompt_config
        )

    async def __call__(self, chat_history: list[Message], current_order: dict) -> str:
        """Executes order intent classification flow using Semantic Kernel."""
        current_order = LLMOrder.model_validate(current_order).model_dump()
        user_message = chat_history[-1].content
        
        # Prepare variables for the prompt using KernelArguments
        variables = KernelArguments(
            user_message=user_message,
            current_order=current_order
        )
        
        # Invoke the prompt function
        response = await self.kernel.invoke(
            plugin_name=self.PROMPT_PLUGIN_NAME,
            function_name=self.PROMPT_FUNCTION_NAME,
            arguments=variables
        )
        msg = str(response)
        return "order" if "<yes>" in msg.lower() else "conversation"
