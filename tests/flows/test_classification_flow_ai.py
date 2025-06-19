import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from pathlib import Path
from streaming_ordering_chatbot.api.flows.schemas import LLMOrder
from streaming_ordering_chatbot.api.models import Message, IntentResponse
from promptflow.entities import AzureOpenAIModelConfiguration
from your_module import OrderIntentFlow  # Replace 'your_module' with the actual module name

@pytest.fixture
def model_config():
    return AzureOpenAIModelConfiguration(
        api_
    )

@pytest.fixture
def order_intent_flow(model_config):
    return OrderIntentFlow(model_config)

@pytest.fixture
def sample_user_message():
    return "I want to modify my order."

@pytest.fixture
def sample_chat_history():
    return [Message(role="user", content="Hello")]

@pytest.fixture
def sample_current_order():
    return {"order_id": "1234", "items": [{"name": "item1", "quantity": 1}]}

@pytest.mark.asyncio
async def test_order_intent_flow_initialization(order_intent_flow, model_config):
    assert order_intent_flow.model_config == model_config
    assert order_intent_flow.prompt_path == Path(__file__).parent.joinpath("prompts/order_intent.prompty")
    assert order_intent_flow.model["configuration"] == model_config
    assert order_intent_flow.model["parameters"]["max_tokens"] == 300

@pytest.mark.asyncio
async def test_order_intent_flow_call(order_intent_flow, sample_user_message, sample_chat_history, sample_current_order):
    with patch("your_module.Prompty.load") as mock_load:
        mock_intent_flow = AsyncMock()
        mock_load.return_value = mock_intent_flow
        mock_intent_flow.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="<yes>"))])

        response = await order_intent_flow(sample_user_message, sample_chat_history, sample_current_order)
        assert isinstance(response, IntentResponse)
        assert response.modify_order is True

@pytest.mark.asyncio
async def test_order_intent_flow_no_modification(order_intent_flow, sample_user_message, sample_chat_history, sample_current_order):
    with patch("your_module.Prompty.load") as mock_load:
        mock_intent_flow = AsyncMock()
        mock_load.return_value = mock_intent_flow
        mock_intent_flow.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="No changes needed."))])

        response = await order_intent_flow(sample_user_message, sample_chat_history, sample_current_order)
        assert isinstance(response, IntentResponse)
        assert response.modify_order is False

@pytest.mark.asyncio
async def test_order_intent_flow_invalid_current_order(order_intent_flow, sample_user_message, sample_chat_history):
    invalid_current_order = {"invalid_key": "invalid_value"}

    with patch("your_module.LLMOrder.model_validate", side_effect=ValueError("Invalid order")), \
         pytest.raises(ValueError):
        await order_intent_flow(sample_user_message, sample_chat_history, invalid_current_order)

@pytest.mark.asyncio
async def test_order_intent_flow_empty_user_message(order_intent_flow, sample_chat_history, sample_current_order):
    empty_user_message = ""

    with patch("your_module.Prompty.load") as mock_load:
        mock_intent_flow = AsyncMock()
        mock_load.return_value = mock_intent_flow
        mock_intent_flow.return_value = MagicMock(choices=[MagicMock(message=MagicMock(content="No changes needed."))])

        response = await order_intent_flow(empty_user_message, sample_chat_history, sample_current_order)
        assert isinstance(response, IntentResponse)
        assert response.modify_order is False