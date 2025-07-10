from typing import Literal, Optional

from pydantic import BaseModel

from streaming_ordering_chatbot.api.flows.schemas_generalized import LLMItem, LLMOrder


class LLMConfig(BaseModel):
    personality: Optional[str] = None
    deployment: Optional[str] = None


class Message(BaseModel):
    role: Literal["user", "assistant", "system"]
    content: str
    filtered: bool = False


class Preferences(BaseModel):
    numPeople: int = 0
    eventDescription: str = ""
    foodPreferences: list[str] = []


class OrderState(BaseModel):
    chat_history: list[Message]
    order: LLMOrder
    preferences: Optional[Preferences] = None
    isOrderNew: bool = False
    isOrderChanged: bool = False


class ScreeningResponse(BaseModel):
    redacted_message: str
    failed_categories: list[str]
    intent: str


class ScreenData(BaseModel):
    message: str
    chat_history: list[Message]
    current_order: LLMOrder


class ExpectedPreferences(BaseModel):
    numPeople: Optional[int] = None
    eventDescription: Optional[str] = None
    foodPreferences: Optional[list[str]] = None


class ExpectedNumItems(BaseModel):
    burgers: Optional[int] = None
    cheeseburgers: Optional[int] = None
    hamburgers: Optional[int] = None
    vegetarianBurgers: Optional[int] = None
    drinks: Optional[int] = None
    fries: Optional[int] = None


class ExpectedRecommendation(BaseModel):
    order: Optional[LLMOrder] = None
    preferences: Optional[Preferences] = None
    expectedNumItems: Optional[ExpectedNumItems] = None
    expectedItems: Optional[list[LLMItem]] = []
    recommendationCreated: Optional[bool] = None
