import asyncio
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from openai import AzureOpenAI

# Load environment variables from .env file
load_dotenv()

# Azure OpenAI configuration
ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

# Validate environment variables
if not all([ENDPOINT, API_KEY, DEPLOYMENT_NAME]):
    raise ValueError("Missing required environment variables. Please check your .env file.")

# Type assertions for static type checking
assert ENDPOINT is not None
assert API_KEY is not None
assert DEPLOYMENT_NAME is not None
ENDPOINT = str(ENDPOINT)
API_KEY = str(API_KEY)
DEPLOYMENT_NAME = str(DEPLOYMENT_NAME)

def read_prompt_template(template_name: str) -> str:
    """Read a prompt template from the prompts directory.
    
    Args:
        template_name (str): Name of the template file (e.g., 'preamble_SK', 'assistant_SK', 'summary_SK')
    
    Returns:
        str: The contents of the prompt template file
    """
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

async def natural_conversation() -> None:
    """Run a natural restaurant ordering conversation with direct LLM output.
    Uses semantic-kernel prompt templates and streams raw model output without post-processing.
    
    The conversation flows through three stages:
    1. Greeting (preamble_SK template) - Initial welcome and menu introduction
    2. Ordering (assistant_SK template) - Menu questions and order taking
    3. Summary (summary_SK template) - Order summary and confirmation
    """
    # Initialize conversation state
    chat_history = []
    current_order = {"items": []}
    is_first_message = True
    
    # Create Azure OpenAI client for direct streaming
    assert ENDPOINT is not None  # Ensure ENDPOINT is not None
    client = AzureOpenAI(
        api_key=API_KEY,
        api_version="2023-12-01-preview",
        azure_endpoint=ENDPOINT
    )
    
    # Load menu content
    menu_path = Path(__file__).parent.joinpath(
        "streaming_ordering_chatbot",
        "api",
        "flows",
        "prompts",
        "menu.txt"
    )
    menu = ""
    try:
        with open(menu_path, 'r', encoding='utf-8') as f:
            menu = f.read()
    except Exception as e:
        menu = "## Menu\n- Burgers\n- Salads\n- Sides\n- Drinks"
        print(f"Using default menu due to error: {e}")
    
    # Load semantic-kernel prompt templates
    preamble_template = read_prompt_template("preamble_SK")
    assistant_template = read_prompt_template("assistant_SK")
    summary_template = read_prompt_template("summary_SK")
    
    # Inject menu into assistant template
    assistant_template = assistant_template.replace("{{ $menu }}", menu)
    
    print("\nStarting Restaurant Ordering Conversation")
    print("Commands: 'exit' to end, 'summary' for chat summary, 'done' when finished ordering")
    print("-" * 80)
    
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
            chat_history.append({"role": "user", "content": user_input})
            
            print("\nAssistant:", end=' ', flush=True)
            
            # Choose appropriate template based on conversation state
            if user_input.lower() == 'summary' and len(chat_history) > 1:
                # Generate conversation summary
                system_prompt = summary_template
            elif is_first_message:
                # First message - use preamble for greeting
                system_prompt = preamble_template
                is_first_message = False
            else:
                # Regular ordering interaction
                system_prompt = assistant_template
                if current_order.get("items"):
                    # Include current order in prompt if items exist
                    order_str = json.dumps(current_order, indent=2)
                    system_prompt = system_prompt.replace("{{ $current_order }}", order_str)
                else:
                    system_prompt = system_prompt.replace("{{ $current_order }}", "No items in order yet")
            
            # Format chat history for template
            chat_history_str = "\n".join([
                f"{msg['role'].title()}: {msg['content']}"
                for msg in chat_history[:-1]  # Exclude current message
            ])
            
            # Update template placeholders
            system_prompt = system_prompt.replace("{{$chat_history}}", chat_history_str)
            
            # Create messages for API call
            api_messages = [
                {"role": "system", "content": system_prompt},
                *chat_history  # Add all chat history messages
            ]
            
            # Get streaming response directly from OpenAI
            completion = client.chat.completions.create(
                model=str(DEPLOYMENT_NAME),
                messages=api_messages,
                temperature=0.7,
                stream=True,
                max_tokens=1000
            )
            
            # Stream tokens without post-processing
            response = ""
            for chunk in completion:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                    token = chunk.choices[0].delta.content
                    print(token, end="", flush=True)
                    response += token
            
            # Add response to chat history
            chat_history.append({"role": "assistant", "content": response})
            print()
            
            # End conversation if order is complete
            if user_input.lower() == 'done' and len(current_order.get("items", [])) > 0:
                print("\nOrder completed! Thank you for dining with us.")
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
            # Validate Azure OpenAI settings
            if not ENDPOINT or not API_KEY:
                raise EnvironmentError("Missing required Azure OpenAI credentials")
            
            await natural_conversation()
            
        except EnvironmentError as e:
            print(f"\nConfiguration Error: {str(e)}")
            print("\nPlease ensure you have set these environment variables:")
            print("- AZURE_OPENAI_ENDPOINT")
            print("- AZURE_OPENAI_KEY")
            print("- AZURE_OPENAI_DEPLOYMENT_NAME (optional, defaults to 'gpt-4o')")
        except Exception as e:
            print(f"\nAn unexpected error occurred: {str(e)}")
            print("If the error persists, check your network connection and Azure OpenAI service status.")
    
    asyncio.run(main())
