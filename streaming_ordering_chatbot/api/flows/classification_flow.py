from pathlib import Path

from promptflow.core import AzureOpenAIModelConfiguration, Prompty
from promptflow.tracing import trace

from streaming_ordering_chatbot.api.flows.schemas import LLMOrder
from streaming_ordering_chatbot.api.models import Message


class OrderIntentFlow:
    def __init__(self, model_config: AzureOpenAIModelConfiguration):
        """Classifies user intent for order modification.

        Args:
            model_config (AzureOpenAIModelConfiguration): Model configuration overrides
        """
        self.model_config = model_config
        self.prompt_path = Path(__file__).parent.joinpath(
            "prompts/order_intent.prompty"
        )
        self.model = {
            "configuration": self.model_config,
            "parameters": {
                "max_tokens": 300,
            },
            "response": "all",
        }

    @trace
    async def __call__(self, chat_history: list[Message], current_order: dict) -> str:
        """Executes order intent classification flow.

        Args:
            chat_history (list[Message]): Chat history
            current_order (dict): Current LLMorder object

        Returns:
            AsyncGenerator[str]:
        """

        current_order = LLMOrder.model_validate(current_order).model_dump()
        intent_flow = Prompty.load(source=self.prompt_path, model=self.model)
        user_message = chat_history[-1].content
        response = intent_flow(user_message=user_message, current_order=current_order)
        msg = response.choices[0].message.content
        return "order" if "<yes>" in msg.lower() else "conversation"
