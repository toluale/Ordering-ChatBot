import asyncio
import json
import os
from typing import Dict, List, Optional
from streaming_ordering_chatbot.api.flows.conversation_flows_SK import (
    PreambleFlowSK,
    SummaryFlowSK,
    OrderAssistantFlowSK
)
from streaming_ordering_chatbot.api.models import Message

ENDPOINT = "https://t-toluale-1040-resource.openai.azure.com/"
API_KEY = "8cfBQF1HE4qzxIn5VapNbWeqhqpYIR6OnHq0zXvxp3gVOz3YC2uOJQQJ99BFACHYHv6XJ3w3AAAAACOGGDMG"
DEPLOYMENT_NAME = "gpt-4o" 
''''
# Azure OpenAI configuration - Get from environment variables
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
if not ENDPOINT:
    raise ValueError("AZURE_OPENAI_ENDPOINT environment variable is required")

API_KEY = os.getenv("AZURE_OPENAI_KEY")
if not API_KEY:
    raise ValueError("AZURE_OPENAI_KEY environment variable is required")

DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4o")
'''
assert isinstance(ENDPOINT, str)
assert isinstance(API_KEY, str)
assert isinstance(DEPLOYMENT_NAME, str)

# Load brand configurations
BRAND_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "streaming_ordering_chatbot",
    "resources",
    "brand_configs.json"
)

try:
    with open(BRAND_CONFIG_PATH, 'r') as f:
        BRAND_CONFIGS = json.load(f)
except Exception as e:
    print(f"Warning: Could not load brand configs: {e}")
    BRAND_CONFIGS = {}

# Created to test different conversation flows
# It should be modified to follow the same structure as the classification flow
def create_flow(flow_type: str = "order") -> PreambleFlowSK | SummaryFlowSK | OrderAssistantFlowSK:
    """Create a flow object based on the flow type."""
    if ENDPOINT is None or API_KEY is None:
        raise ValueError("Azure OpenAI environment variables are not set")
    
    if flow_type == "preamble":
        return PreambleFlowSK(ENDPOINT, API_KEY, DEPLOYMENT_NAME)
    elif flow_type == "summary":
        return SummaryFlowSK(ENDPOINT, API_KEY, DEPLOYMENT_NAME)
    else:  # default to order
        return OrderAssistantFlowSK(ENDPOINT, API_KEY, DEPLOYMENT_NAME)

async def interactive_conversation(
    flow_type: str = "order"
):
    """Run an interactive conversation with the specified flow type.
    
    This function starts an interactive chat session with the chosen AI flow. You can
    type messages and get responses from the AI. Special commands include:
    - Type 'exit' to end the conversation
    - Type 'summary' to get a summary of the conversation so far
    
    Args:
        flow_type (str): Type of flow to use:
            - "preamble": Initial greeting flow
            - "order": Food ordering assistant (default)
            - "summary": Conversation summarization
    
    Returns:
        None. The function runs interactively and prints responses to the console.
    """
    # Initialize state
    chat_history = []
    current_order = {"items": []}
    response_text = []
    
    # Initialize the appropriate flow
    try:
        flow = create_flow(flow_type)
    except ValueError as e:
        print(f"Error: {e}")
        return
    
    print("\nStarting conversation (type 'exit' to end, 'summary' for chat summary)...")
    
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                break
            
            # Add user message to chat history
            chat_history.append(Message(role="user", content=user_input))
              # Handle summary request
            if user_input.lower() == 'summary':
                print("\nGenerating conversation summary...")
                summary_flow = create_flow("summary")
                print("\nSummary:", end=' ', flush=True)
                
                # Process all chunks from the summary generator
                summary_chunks = []
                async for chunk in summary_flow(chat_history=chat_history):
                    if chunk:
                        print(chunk, end='', flush=True)
                        summary_chunks.append(chunk)
                
                # Join chunks into final summary
                summary_text = "".join(summary_chunks)
                print("\n")
                chat_history.append(Message(role="assistant", content=summary_text))
                continue
          # Get and stream assistant's response
            print("\nAssistant:", end=' ', flush=True)
            response_text.clear()
            
            # Call the appropriate flow and collect all chunks
            if flow_type == "order":
                generator = flow(
                    chat_history=chat_history,
                    current_order=current_order
                )
            else:
                generator = flow(
                    chat_history=chat_history
                )
            
            # Process all chunks from the generator
            response_chunks = []
            async for chunk in generator:
                if chunk:  # Skip empty chunks
                    print(chunk, end='', flush=True)
                    response_chunks.append(chunk)
            
            # Join all chunks into the final response
            full_response = "".join(response_chunks)
            print()
            
            # Add assistant's response to chat history
            chat_history.append(Message(role="assistant", content=full_response))
        except Exception as e:
            print(f"\nError: {e}")

async def test_conversation_flows():
    """Test different conversation flows"""
    
    print("\n=== Testing Preamble Flow ===")
    await interactive_conversation(flow_type="preamble")
    
    print("\n=== Testing Order Assistant (Default) ===")
    await interactive_conversation(flow_type="order")
    
    print("\n=== Testing Summary Flow ===")
    await interactive_conversation(flow_type="summary")

async def run_integrated_conversation_test() -> None:
    """Run an integrated conversation test that flows through all stages:
    1. Greeting (Preamble)
    2. Order placement
    3. Summary
    
    This simulates a complete user journey through the conversation system.
    User will be prompted to enter messages for each stage of the conversation.
    """
    print("\nPlease provide your messages for the conversation test:")
    
    # Get greeting message
    while True:
        greeting_message = input("\nEnter your greeting message: ").strip()
        if greeting_message:
            break
        print("Please enter a greeting message.")
    
    # Get order messages
    print("\nEnter your order messages (one per line).")
    print("Press Enter twice when done.")
    order_messages = []
    while True:
        message = input("\nEnter order message (or press Enter to finish): ").strip()
        if not message and order_messages:  # Empty line and we have messages
            break
        if not message:  # Empty line but no messages yet
            print("Please enter at least one order message.")
            continue
        order_messages.append(message)
    
    print("\n=== Starting Integrated Conversation Test ===\n")
    
    # Initialize shared state
    chat_history = []
    current_order = {"items": []}
    
    # 1. Preamble Flow
    print("Stage 1: Initial Greeting")
    print(f"\nUser: {greeting_message}")
    preamble_flow = create_flow("preamble")
    
    # Add user message and get response
    chat_history.append(Message(role="user", content=greeting_message))
    print("\nAssistant:", end=' ', flush=True)
    
    async for chunk in preamble_flow(chat_history=chat_history):
        if chunk:
            print(chunk, end='', flush=True)
    print("\n")
    
    # 2. Order Flow
    print("\nStage 2: Order Placement")
    order_flow = create_flow("order")
    
    for message in order_messages:
        print(f"\nUser: {message}")
        chat_history.append(Message(role="user", content=message))
        print("\nAssistant:", end=' ', flush=True)
        
        async for chunk in order_flow(chat_history=chat_history, current_order=current_order):
            if chunk:
                print(chunk, end='', flush=True)
        print("\n")
    
    # 3. Summary Flow
    print("\nStage 3: Conversation Summary")
    summary_flow = create_flow("summary")
    print("\nGenerating final summary...")
    
    async for chunk in summary_flow(chat_history=chat_history):
        if chunk:
            print(chunk, end='', flush=True)
    print("\n")
    
    print("\n=== Integrated Conversation Test Complete ===\n")

async def natural_conversation_flow() -> None:
    """Run a natural conversation flow where the assistant responds contextually to user input.
    The conversation flows naturally through greeting, ordering, and summary based on user interactions.
    The assistant will:
    1. Start with a greeting and respond to initial context
    2. Handle orders when the user wants to place them
    3. Provide order validation and confirmation
    4. Generate a summary when the order is complete
    """
    # Initialize shared state
    chat_history = []
    current_order = {"items": []}
    
    # Initialize all flows we might need
    preamble_flow = create_flow("preamble")
    order_flow = create_flow("order")
    summary_flow = create_flow("summary")
    
    # Track conversation state
    is_order_complete = False
    
    print("\nStarting conversation (type 'exit' to end, 'done' when finished ordering)...")
    
    # Start with greeting phase
    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()
            
            if user_input.lower() == 'exit':
                break
                
            # Add user message to chat history
            chat_history.append(Message(role="user", content=user_input))
            
            print("\nAssistant:", end=' ', flush=True)
            
            if user_input.lower() == 'done' and len(current_order.get("items", [])) > 0:
                # User has finished ordering, generate final summary
                is_order_complete = True
                print("\nGenerating order summary...")
                async for chunk in summary_flow(chat_history=chat_history):
                    if chunk:
                        print(chunk, end='', flush=True)
                print("\n")
                break
            elif len(chat_history) == 1:
                # First message - use preamble flow
                async for chunk in preamble_flow(chat_history=chat_history):
                    if chunk:
                        print(chunk, end='', flush=True)
            else:
                # Use order flow for all subsequent messages
                async for chunk in order_flow(chat_history=chat_history, current_order=current_order):
                    if chunk:
                        print(chunk, end='', flush=True)
            print()
            
        except Exception as e:
            print(f"\nError: {e}")
    
    if not is_order_complete and len(current_order.get("items", [])) > 0:
        # Generate final summary if we have an order but didn't get one yet
        print("\nGenerating final order summary...")
        async for chunk in summary_flow(chat_history=chat_history):
            if chunk:
                print(chunk, end='', flush=True)
        print("\n")

if __name__ == "__main__":
    print("\nChoose a conversation mode:")
    print("1. Natural Conversation Flow")
    print("\nOr select a test mode:")
    print("2. Run all conversation flow tests")
    print("3. Run integrated conversation test")
    print("\nOr select a specific interactive mode:")
    print("4. Interactive Preamble conversation")
    print("5. Interactive Order Assistant")
    print("6. Interactive Summary conversation")
    
    choice = input("\nEnter your choice (1-5): ")
    async def main():
        try:
            choice_num = int(choice)
            if choice_num < 1 or choice_num > 6:
                raise ValueError()
            
            if choice_num == 1:
                await natural_conversation_flow()
            elif choice_num == 2:
                await test_conversation_flows()            
            elif choice_num == 3:
                await run_integrated_conversation_test()
            elif choice_num == 4:
                await interactive_conversation(flow_type="preamble")
            elif choice_num == 5:
                await interactive_conversation(flow_type="order")
            elif choice_num == 6:
                await interactive_conversation(flow_type="summary")
        except ValueError:
            print("Invalid choice. Please run the script again and select a number between 1 and 5.")
        except Exception as e:
            print(f"An error occurred: {e}")
            if ENDPOINT is None or API_KEY is None:
                print("\nMake sure you have set these environment variables:")
                print("- AZURE_OPENAI_ENDPOINT")
                print("- AZURE_OPENAI_KEY")
                print("- AZURE_OPENAI_DEPLOYMENT_NAME (optional, defaults to 'gpt-4o')")
    
    asyncio.run(main())