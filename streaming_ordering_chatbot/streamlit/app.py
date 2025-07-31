import asyncio
import json
from pathlib import Path
from uuid import uuid4

import httpx
import streamlit as st
import os

# Environment-configurable API base
API_BASE_URL = os.getenv("STREAMLIT_API_BASE_URL", "http://localhost:8000")

# API Endpoints
ORDER_ENDPOINT = f"{API_BASE_URL}/order"
PREAMBLE_ENDPOINT = f"{API_BASE_URL}/preamble"
SUMMARY_ENDPOINT = f"{API_BASE_URL}/summary"
SCREENING_ENDPOINT = f"{API_BASE_URL}/screen"
CONVERSATION_ENDPOINT = f"{API_BASE_URL}/assistant"

# Frontend resources
BASE_DIR = Path(__file__).resolve().parent.parent
TONE_FILES = {
    "Casual": BASE_DIR.joinpath("resources/casual.txt"),
    "GenZ": BASE_DIR.joinpath("resources/genZ.txt"),
}

TONE_CHOICES = ["Default"] + list(TONE_FILES.keys())

# Add this mapping to convert UI choices to API values
TONE_TO_STYLE_MAPPING = {
    "Default": "default",
    "Casual": "casual", 
    "GenZ": "genz"
}


MODEL_CHOICES = {
    "GPT-4o": "gpt-4o",
    "GPT-4.1": "gpt-4.1"
}

async def fetch_stream(url, container, json_data):
    async with httpx.AsyncClient(timeout=30) as client:
        headers = {
            "brand-session-id": st.session_state.session_id,
            "request-id": str(uuid4()),
        }
        async with client.stream(
            "POST", url, json=json_data, headers=headers
        ) as response:
            current_content = ""
            async for line in response.aiter_lines():
                if line:
                    if "<REDACTED" in line:
                        # All previous content is sent with redacted information
                        current_content = line
                    else:
                        current_content += line + "\n"
                    
                    # Post-process the content to ensure proper markdown formatting
                    container.markdown(current_content, unsafe_allow_html=True)
    return current_content


async def fetch_order(current_order, container, items_list, chat_history, state):
    order_obj = ""
    first_line = True
    order_finished = False
    desc = []  # Initialize desc at the beginning of the function
    order = None  # Initialize order as well
    
    async with httpx.AsyncClient(timeout=30) as client:
        headers = {
            "brand-session-id": st.session_state.session_id,
            "request-id": str(uuid4()),
        }
        async with client.stream(
            "POST",
            ORDER_ENDPOINT,
            json={
                "state": {
                    "chat_history": chat_history,
                    "order": current_order,
                },
                "config": {
                    "deployment": MODEL_CHOICES[st.session_state.selected_model]
                },
            },
            headers=headers,
        ) as response:
            async for line in response.aiter_lines():
                if line:
                    if first_line:
                        order_obj += line
                        first_line = False
                    elif order_finished:
                        try:
                            state.llm_order = json.loads(line)["LLMOrder"]
                        except (json.JSONDecodeError, KeyError) as e:
                            st.warning(f"Could not parse final order: {e}")
                    else:
                        order_obj += line
                        if line == "]}":
                            order_finished = True
                            try:
                                order = json.loads(order_obj)
                            except json.JSONDecodeError:
                                st.error("Could not parse order JSON")
                                return "Error parsing order", None
                        else:
                            try:
                                order = json.loads(
                                    order_obj + "]}"
                                )  # Attempt to decode the JSON
                            except json.JSONDecodeError:
                                continue  # Continue instead of return None to keep streaming
                        
                        # Safely extract descriptions (only if we have a valid order)
                        if order:
                            desc = []  # Reset desc for this iteration
                            order_items = order.get("order", [])
                            
                            for item in order_items:
                                # Try different ways to get item description
                                description = None
                                
                                # Method 1: Check for 'description' field
                                if "description" in item:
                                    description = item["description"]
                                
                                # Method 2: Build from name and quantity
                                elif "name" in item:
                                    name = item["name"]
                                    quantity = item.get("quantity", 1)
                                    description = f"{quantity}x {name}"
                                    
                                    # Add size/options if available
                                    if "size" in item:
                                        description += f" ({item['size']})"
                                
                                # Method 3: Fallback to string representation
                                else:
                                    description = str(item)
                                
                                if description:
                                    desc.append(description)
                            
                            # Update UI
                            container.markdown(order)
                            if desc:
                                items_list.markdown("- " + "\n - ".join(desc))
                            else:
                                items_list.markdown("- No items in order")
                            
    return ("- " + "\n - ".join(desc)) if desc else "No items", order

async def fetch_brand_info():
    """Fetch brand information from the API."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            # Try to get conversation styles first (which includes brand info)
            response = await client.get(f"{API_BASE_URL}/conversation-styles")
            if response.status_code == 200:
                data = response.json()
                # Extract brand name from the response if available
                return data.get("current_brand", "Restaurant")
            else:
                return "Restaurant"  # Fallback
    except Exception as e:
        st.warning(f"Could not connect to API: {e}")
        return "Restaurant"

def load_prompt(prompt_path):
    """
    Load prompt from a file.

    Parameters:
    - prompt_path (str): The path to the file containing the prompt.

    Returns:
    - str: The content of the file as a string.
    """
    with open(prompt_path, "r") as file:
        return file.read()

async def generate_initial_greeting():
    """Generate an initial greeting using the brand personality system."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            headers = {
                "brand-session-id": st.session_state.session_id,
                "request-id": str(uuid4()),
            }
            
            # Create a minimal chat history with a greeting trigger
            initial_chat_history = [
                {
                    "role": "user",
                    "content": "Hello",
                    "filtered": False
                }
            ]
            
            json_data = {
                "chat_history": initial_chat_history,
                "config": {
                    "conversation_style": "default",  # Use default style for initial greeting
                    "deployment": "gpt-4o",  # Use a default model
                },
            }
            
            response = await client.post(
                PREAMBLE_ENDPOINT,
                json=json_data,
                headers=headers
            )
            
            if response.status_code == 200:
                # For streaming responses, we need to collect all chunks
                greeting = ""
                async for line in response.aiter_lines():
                    if line:
                        greeting += line
                return greeting.strip()
            else:
                # Fallback if API fails
                return f"Welcome to {st.session_state.brand_name}! How can I help you today?"
                
    except Exception as e:
        st.warning(f"Could not generate dynamic greeting: {e}")
        # Fallback to basic greeting
        return f"Welcome to {st.session_state.brand_name}! How can I help you today?"

async def main():

    # st.set_page_config(layout="wide")
    # Initialize brand name from API
    if "brand_name" not in st.session_state:
        st.session_state.brand_name = await fetch_brand_info()

    # Initialize chat history
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "llm_order" not in st.session_state:
        st.session_state.llm_order = {"items": []}
    if "messages" not in st.session_state:
        initial_greeting = await generate_initial_greeting()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": initial_greeting,
                "filtered": False,
            }
        ]
    if "order" not in st.session_state:
        st.session_state.order = {"items": []}
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "GPT-4o"
    if "prompts" not in st.session_state:
        st.session_state.prompts = {k: load_prompt(v) for k, v in TONE_FILES.items()}
        st.session_state.selected_prompt = "Default"

    prompt = st.chat_input("")
    chat_col, cart_col = st.columns([7, 3], gap="large")

    with st.sidebar:
        tone_choice = st.selectbox("Choose a tone", options=TONE_CHOICES)

        if st.button("Apply Tone"):
            st.session_state.selected_prompt = tone_choice
            st.success(f"{tone_choice} tone applied to the system message!")

        model_choice = st.selectbox("Select Model", options=MODEL_CHOICES)

        if st.button("Select Model"):
            st.session_state.selected_model = model_choice
            st.success(f"Model updated to {model_choice}!")

    with cart_col:
        cart = st.empty()

    with chat_col:
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display chat messages from history on app rerun
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt:
            # Display user message in chat message container
            last_user = st.chat_message("user")
            last_user_content = last_user.empty()
            last_user_content.markdown(prompt)
            # Add user message to chat history

            response = f"Echo: {prompt}"
            # Display assistant response in chat message container
            last_asst = st.chat_message("assistant")
            chat_history = [
                message
                for message in st.session_state.messages
                if not message.get("filtered")
            ]
            with last_asst:
                async with httpx.AsyncClient(timeout=30) as client:
                    headers = {
                        "brand-session-id": st.session_state.session_id,
                        "request-id": str(uuid4()),
                    }
                    data = {
                        "message": prompt,
                        "chat_history": chat_history,
                        "current_order": st.session_state.llm_order,
                    }
                    res = await client.post(
                        SCREENING_ENDPOINT, json=data, headers=headers
                    )
                    screening_result = res.json()
                if len(screening_result["failed_categories"]) > 0:
                    text = "I'm sorry, I can't process your request. Could you please try again?"
                    last_asst.markdown(text)
                    user_message = (
                        "<Redacted for content safety: "
                        + ", ".join(screening_result["failed_categories"])
                        + ">"
                    )
                    last_user_content.markdown(user_message)
                    # Edit last user message
                    st.session_state.messages.append(
                        {"role": "user", "content": user_message, "filtered": True}
                    )
                    st.session_state.messages.append(
                        {"role": "assistant", "content": text, "filtered": True}
                    )
                else:

                    last_user_content.markdown(screening_result["redacted_message"])
                    user_message = {
                        "role": "user",
                        "content": screening_result["redacted_message"],
                        "filtered": False,
                    }

                    chat_history.append(user_message)
                    if screening_result["intent"] == "order":
                        # Remove preamble = st.empty()
                        items_list = st.empty()
                        summary = st.empty()
                        
                        order = asyncio.create_task(
                            fetch_order(
                                {"items": []},
                                cart,
                                items_list,
                                chat_history,
                                st.session_state,
                            )
                        )
                        json_data = {
                            "chat_history": chat_history,
                            "config": {
                                "conversation_style": TONE_TO_STYLE_MAPPING.get(
                                    st.session_state.selected_prompt, "default"
                                ),
                                "deployment": MODEL_CHOICES[
                                    st.session_state.selected_model
                                ],
                            },
                        }

                        # Wait for order processing to complete
                        order_result = await order
                        
                        # Add order info to chat history for summary generation
                        if order_result[1] is None:
                            order_info = "Failed to fetch order, item might have not existed"
                        else:
                            order_info = order_result[0]
                        
                        # Add order info to chat history
                        chat_history.append({
                            "role": "assistant", 
                            "content": f"Order processed: {order_info}"
                        })
                        
                        # Generate summary response
                        summary_response = await fetch_stream(
                            SUMMARY_ENDPOINT,
                            summary,
                            json_data,
                        )
                        assistant_message = {
                            "role": "assistant",
                            "content": summary_response,
                        }
                    else:
                        conversation = st.empty()
                        response = await fetch_stream(
                            CONVERSATION_ENDPOINT,
                            conversation,
                            {
                                "chat_history": chat_history,
                                "current_order": st.session_state.llm_order,
                                "config": {
                                    "conversation_style": TONE_TO_STYLE_MAPPING.get(
                                        st.session_state.selected_prompt, "default"
                                    ),
                                    "deployment": MODEL_CHOICES[
                                        st.session_state.selected_model
                                    ],
                                },
                            },
                        )
                        assistant_message = {
                            "role": "assistant",
                            "content": response,
                        }
                    st.session_state.messages.append(user_message)
                    st.session_state.messages.append(assistant_message)


if __name__ == "__main__":
    asyncio.run(main())
