import asyncio
import json
import os
import logging
from pathlib import Path
from typing import Optional, Dict
from dotenv import load_dotenv

from streaming_ordering_chatbot.api.flows import (PreambleFlowSK, OrderAssistantFlowSK, SummaryFlowSK)
from streaming_ordering_chatbot.api.models import Message

# Load environment variables from .env file
load_dotenv()

# Set up logging
logger = logging.getLogger(__name__)

def get_validated_config() -> tuple[str, str, str, str]:
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    brand_name = os.getenv("BRAND_NAME")  
    
    if not all([endpoint, api_key, deployment_name]):
        raise ValueError("Missing required environment variables. Please check your .env file.")
    
    endpoint = str(endpoint)
    api_key = str(api_key)
    deployment_name = str(deployment_name)
    brand_name = str(brand_name)
    
    return endpoint, api_key, deployment_name, brand_name

def read_prompt_template(template_name: str) -> str:
    """Read a prompt template from the prompts directory and return its content."""
    template_path = Path(__file__).parent.joinpath(
        "streaming_ordering_chatbot",
        "api",
        "flows",
        "prompts",
        f"{template_name}.prompty"
    )
    
    try:
        with open(template_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Warning: Could not load {template_name} template: {e}")
        return ""

def initialize_conversation_flows(
    endpoint: str,
    api_key: str,
    deployment_name: str,
    brand_name: str
):
    """Initialize conversation flows with parameters."""
    return (
        PreambleFlowSK(endpoint, api_key, deployment_name, brand_name=brand_name),
        OrderAssistantFlowSK(endpoint, api_key, deployment_name, brand_name=brand_name),
        SummaryFlowSK(endpoint, api_key, deployment_name, brand_name=brand_name)
    )

async def natural_conversation() -> None:
    """Replicating natural restaurant ordering conversation using ConversationFlowSK classes with brand config incorporated.
    1. Greeting (PreambleFlowSK) - Initial welcome and menu introduction
    2. Ordering (OrderAssistantFlowSK) - Menu questions and order taking
    3. Summary (SummaryFlowSK) - Order summary and confirmation
    """
    endpoint, api_key, deployment_name, brand_name = get_validated_config()
    preamble_flow, order_ass_flow, summary_flow = initialize_conversation_flows(
        endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment_name,
        brand_name=brand_name
    )
    
    # Initialize conversation state
    chat_history: list[Message] = []
    current_order = {"items": []}
    is_first_message = True
    
    print("\nStarting Restaurant Ordering Conversation")
    print(f"Using brand personality: {brand_name}")
    print("Commands: 'exit' to end, 'summary' for chat summary, 'done' when finished ordering")
    print("-" * 80)
    #this return a response for every user input
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if not user_input:
                print("Please enter a message.")
                continue
                
            if user_input.lower() == 'exit':
                break
            
            # Add user message to chat history
            chat_history.append(Message(role="user", content=user_input))
            
            print("\nAssistant:", end=' ', flush=True)
              # using appropriate flow based on conversation state and get response
            response = ""
            try:
                if user_input.lower() == 'summary' and len(chat_history) > 1:
                    # conversation summary
                    flow = summary_flow
                    async for token in flow(chat_history, current_order):
                        response += token
                        print(token, end='', flush=True)
                elif is_first_message:
                    # First message - use preamble for greeting
                    flow = preamble_flow
                    async for token in flow(chat_history, current_order):
                        response += token
                        print(token, end='', flush=True)
                    is_first_message = False
                    # If first message is about menu/order, immediately follow up with order ass flow
                    if any(keyword in user_input.lower() for keyword in ['menu', 'order', 'food', 'drink', 'what', 'have', 'give']):
                        flow = order_ass_flow
                        async for token in flow(chat_history, current_order):
                            response += token
                            print(token, end='', flush=True)
                else:
                    # Regular ordering interaction
                    flow = order_ass_flow
                    async for token in flow(chat_history, current_order):
                        response += token
                        print(token, end='', flush=True)
                
                # Add response to chat history
                if response:
                    chat_history.append(Message(role="assistant", content=response))
                print()
            except Exception as e:
                logger.error(f"Error in conversation flow: {e}")
                print(f"\nI apologize, but I encountered an error. Let me try to help you again.")
            
            # End conversation if order is complete
            if user_input.lower() == 'done' and len(current_order.get("items", [])) > 0:
                print("\nOrder completed! Thank you for patronizing us.")
                break
                
        except Exception as e:
            print(f"\nError: {e}")
            print("Trying to continue with conversation...")
            continue

if __name__ == "__main__":
    print("\nWelcome to the Restaurant Ordering Assistant")
    print("This chatbot will help you explore our menu and place your order.")
    
    async def main():
        try:
            await natural_conversation()
            
        except ValueError as e:
            print(f"\nConfiguration Error: {str(e)}")
            print("\nPlease ensure you have set these environment variables:")
            print("- AZURE_OPENAI_ENDPOINT")
            print("- AZURE_OPENAI_API_KEY")
            print("- AZURE_OPENAI_DEPLOYMENT_NAME")
            print("- BRAND_NAME (optional)")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {str(e)}")
            
    asyncio.run(main())
