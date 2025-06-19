import os

import pytest
from promptflow.core import AzureOpenAIModelConfiguration

from streaming_ordering_chatbot.api.flows.order_flow import OrderFlow

flow_configuration = AzureOpenAIModelConfiguration(
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_key=os.environ["AZURE_API_KEY"],
    api_version=os.environ["AZURE_API_VERSION"],
    azure_deployment=os.environ["AZURE_DEPLOYMENT_NAME"],
)


class TestOrderFlow:
    @pytest.fixture
    def order_flow(self):
        return OrderFlow(model_config=flow_configuration)

    def test_order_flow_init(self, order_flow):
        assert order_flow.model_config == flow_configuration
        assert hasattr(order_flow, "tools")
        assert hasattr(order_flow, "menu")

    @pytest.mark.asyncio
    async def test_order_creation(self, order_flow):
        chat_history = "I would like to order hamburger and cola."
        current_order = {"items": []}
        res = order_flow(chat_history, current_order)
        response = ""
        async for item in res:
            response += item
        assert response == "Order created successfully."
