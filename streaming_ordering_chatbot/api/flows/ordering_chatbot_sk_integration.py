import asyncio
import os
from typing import List, Dict, Optional, AsyncGenerator
from dotenv import load_dotenv

from streaming_ordering_chatbot.api.flows.classification_flow_SK import OrderIntentFlowSK
from streaming_ordering_chatbot.api.flows.conversation_flows_SK import (
    PreambleFlowSK,
    OrderAssistantFlowSK,
    SummaryFlowSK
)
from streaming_ordering_chatbot.api.flows.order_flow_SK import OrderFlowSK
from streaming_ordering_chatbot.api.models import Message

# Load environment variables
load_dotenv()

def get_required_env_var(name: str) -> str:
    """Get required environment variable."""
    value = os.getenv(name)
    if not value:
        raise ValueError(f"{name} environment variable is not set. Please set it in your .env file.")
    return value

# Azure OpenAI configuration
ENDPOINT = get_required_env_var("AZURE_OPENAI_ENDPOINT")
API_KEY = get_required_env_var("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = get_required_env_var("AZURE_OPENAI_DEPLOYMENT_NAME")


class OrderingChatBotSK:
    """Complete ordering chatbot implementation using Semantic Kernel."""
    
    def __init__(self, brand_name: Optional[str] = None, conversation_style: str = "default"):
        """Initialize the chatbot with all required flows.
        
        Args:
            brand_name: Optional brand name for personalization. If not provided, will use RESTAURANT_BRAND or BRAND_NAME from environment
            conversation_style: Conversation style (default, friendly, professional, etc.)
        """
        # Get brand name from parameter or environment variables
        if brand_name:
            self.brand_name = brand_name
        else:
            # Try RESTAURANT_BRAND first, then BRAND_NAME for compatibility
            self.brand_name = os.getenv("RESTAURANT_BRAND") or os.getenv("BRAND_NAME")
            if not self.brand_name:
                raise ValueError(
                    "Brand name must be provided either as parameter or through "
                    "RESTAURANT_BRAND or BRAND_NAME environment variables"
                )
        if not brand_name:
            raise ValueError("Brand name is required for OrderingChatBotSK")
        
        self.brand_name = brand_name
        self.conversation_style = conversation_style
        
        # Initialize all flows
        self.classification_flow = OrderIntentFlowSK(ENDPOINT, API_KEY, DEPLOYMENT_NAME)
        self.preamble_flow = PreambleFlowSK(ENDPOINT, API_KEY, DEPLOYMENT_NAME, brand_name, conversation_style)
        self.conversation_flow = OrderAssistantFlowSK(ENDPOINT, API_KEY, DEPLOYMENT_NAME, brand_name, conversation_style)
        self.order_flow = OrderFlowSK(ENDPOINT, API_KEY, DEPLOYMENT_NAME, brand_name)
        self.summary_flow = SummaryFlowSK(ENDPOINT, API_KEY, DEPLOYMENT_NAME, brand_name, conversation_style)
        
        print(f"OrderingChatBotSK initialized for {self.brand_name}")
    
    async def classify_intent(self, chat_history: List[Message], current_order: Dict) -> str:
        """Classify user intent as 'order' or 'conversation'.
        
        Args:
            chat_history: List of chat messages
            current_order: Current order state
            
        Returns:
            Classification result ('order' or 'conversation')
        """
        try:
            intent = await self.classification_flow(chat_history, current_order)
            print(f"Intent classified as: {intent}")
            return intent
        except Exception as e:
            print(f"Error in intent classification: {e}")
            return "conversation"  # Default to conversation on error
    
    async def handle_conversation(
        self, 
        chat_history: List[Message], 
        current_order: Dict,
        delay: float = 0.05
    ) -> AsyncGenerator[str, None]:
        """Handle conversation using conversation flow.
        
        Args:
            chat_history: List of chat messages
            current_order: Current order state
            delay: Streaming delay
            
        Yields:
            Conversation response
        """
        try:
            # Determine if this is a greeting/preamble
            if len(chat_history) <= 1:
                # Use preamble flow for initial greeting
                async for chunk in self.preamble_flow(chat_history, current_order, delay):
                    yield chunk
            else:
                # Use conversation flow for ongoing conversation
                async for chunk in self.conversation_flow(chat_history, current_order, delay):
                    yield chunk
                    
        except Exception as e:
            print(f"Error in conversation handling: {e}")
            yield f"I apologize, but I encountered an error. Please try again."
    
    async def handle_order(
        self, 
        chat_history: List[Message], 
        current_order: Dict,
        delay: float = 0.05
    ) -> AsyncGenerator[str, None]:
        """Handle order processing using order flow.
        
        Args:
            chat_history: List of chat messages
            current_order: Current order state
            delay: Streaming delay
            
        Yields:
            Order processing response
        """
        try:
            async for chunk in self.order_flow(chat_history, current_order, delay):
                yield chunk
                
        except Exception as e:
            print(f"Error in order handling: {e}")
            yield f'{{"error": "Order processing failed: {str(e)}"}}'
    
    async def handle_summary(
        self, 
        chat_history: List[Message], 
        current_order: Dict,
        delay: float = 0.05
    ) -> AsyncGenerator[str, None]:
        """Handle order summary using summary flow.
        
        Args:
            chat_history: List of chat messages
            current_order: Current order state
            delay: Streaming delay
            
        Yields:
            Order summary response
        """
        try:
            async for chunk in self.summary_flow(chat_history, current_order, delay):
                yield chunk
                
        except Exception as e:
            print(f"Error in summary handling: {e}")
            yield f"I apologize, but I couldn't generate your order summary. Please try again."
    
    async def process_message(
        self, 
        chat_history: List[Message], 
        current_order: Dict,
        delay: float = 0.05,
        force_intent: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Process a user message through the complete flow.
        
        Args:
            chat_history: List of chat messages
            current_order: Current order state
            delay: Streaming delay
            force_intent: Optional intent override for testing
            
        Yields:
            Response based on intent classification
        """
        try:
            # Classify intent unless overridden
            if force_intent:
                intent = force_intent
                print(f"Using forced intent: {intent}")
            else:
                intent = await self.classify_intent(chat_history, current_order)
            
            # Route to appropriate handler
            if intent == "order":
                print("Routing to order processing...")
                async for chunk in self.handle_order(chat_history, current_order, delay):
                    yield chunk
            else:
                print("Routing to conversation handling...")
                async for chunk in self.handle_conversation(chat_history, current_order, delay):
                    yield chunk
                    
        except Exception as e:
            print(f"Error in message processing: {e}")
            yield f"I apologize, but I encountered an error processing your message. Please try again."
    
    async def validate_order_item(self, item: Dict) -> bool:
        """Validate a single order item.
        
        Args:
            item: Order item dictionary
            
        Returns:
            True if valid, False otherwise
        """
        try:
            validated_item = await self.order_flow.validate_item(item)
            return validated_item is not None
        except Exception as e:
            print(f"Error validating item: {e}")
            return False


# Example usage and testing
async def test_ordering_chatbot():
    """Test the complete ordering chatbot flow."""
    print("Testing OrderingChatBotSK...")
    
    # Get brand from environment or use default
    brand_name = os.getenv("RESTAURANT_BRAND", "Contoso Restaurant")
    print(f"Testing with brand: {brand_name}")
    
    try:
        # Initialize chatbot
        chatbot = OrderingChatBotSK(brand_name=brand_name, conversation_style="friendly")
        
        # Test cases - customize based on the selected brand
        if "Contoso" in brand_name:
            test_cases = [
                {
                    "name": "Greeting",
                    "messages": [Message(role="user", content="Hello!")],
                    "order": {"items": []},
                    "expected_intent": "conversation"
                },
                {
                    "name": "Burger Order",
                    "messages": [
                        Message(role="user", content="Hello!"),
                        Message(role="assistant", content="Hello! Welcome to Contoso Restaurant!"),
                        Message(role="user", content="I'd like to order a cheeseburger with fries")
                    ],
                    "order": {"items": []},
                    "expected_intent": "order"
                },
                {
                    "name": "General Question",
                    "messages": [
                        Message(role="user", content="What are your hours?")
                    ],
                    "order": {"items": []},
                    "expected_intent": "conversation"
                }
            ]
        elif "Chipotle" in brand_name:
            test_cases = [
                {
                    "name": "Greeting",
                    "messages": [Message(role="user", content="Hello!")],
                    "order": {"items": []},
                    "expected_intent": "conversation"
                },
                {
                    "name": "Burrito Order",
                    "messages": [
                        Message(role="user", content="I want a chicken burrito with extra guac")
                    ],
                    "order": {"items": []},
                    "expected_intent": "order"
                }
            ]
        else:
            # Generic test cases
            test_cases = [
                {
                    "name": "Greeting",
                    "messages": [Message(role="user", content="Hello!")],
                    "order": {"items": []},
                    "expected_intent": "conversation"
                },
                {
                    "name": "Food Order",
                    "messages": [
                        Message(role="user", content="I'd like to place an order")
                    ],
                    "order": {"items": []},
                    "expected_intent": "order"
                }
            ]
        
        # Run tests
        for test_case in test_cases:
            print(f"\nTesting: {test_case['name']}")
            print(f"User message: {test_case['messages'][-1].content}")
            
            # Test classification
            intent = await chatbot.classify_intent(test_case["messages"], test_case["order"])
            print(f"Classified intent: {intent} (expected: {test_case['expected_intent']})")
            
            # Test full processing
            print("Processing message...")
            response_chunks = []
            async for chunk in chatbot.process_message(
                test_case["messages"], 
                test_case["order"], 
                delay=0.01  # Faster for testing
            ):
                response_chunks.append(chunk)
                if len(chunk.strip()) > 0:
                    print(f"Response chunk: {chunk.strip()}")
            
            print(f"Test completed for: {test_case['name']}")
            print("-" * 50)
            
    except Exception as e:
        print(f"Error testing chatbot: {e}")
        print("Make sure:")
        print("1. Environment variables are set correctly")
        print("2. The specified brand is configured with a menu file")
        print("3. Azure OpenAI credentials are valid")


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_ordering_chatbot())
