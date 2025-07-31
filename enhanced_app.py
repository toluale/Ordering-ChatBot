import asyncio
import json
from pathlib import Path
from uuid import uuid4
import time
from datetime import datetime

import httpx
import streamlit as st
import os

from streamlit_evaluation_system_optimized import (
    OptimizedStreamlitEvaluationSystem as StreamlitEvaluationSystem, 
    fetch_stream_with_evaluation, 
    fetch_order_with_evaluation,
    create_evaluation_dashboard,
    initialize_evaluation_system
)

# Environment-configurable API base
API_BASE_URL = os.getenv("STREAMLIT_API_BASE_URL", "http://localhost:8000")

# API Endpoints
ORDER_ENDPOINT = f"{API_BASE_URL}/order"
PREAMBLE_ENDPOINT = f"{API_BASE_URL}/preamble"
SUMMARY_ENDPOINT = f"{API_BASE_URL}/summary"
SCREENING_ENDPOINT = f"{API_BASE_URL}/screen"
CONVERSATION_ENDPOINT = f"{API_BASE_URL}/assistant"

# Frontend resources
BASE_DIR = Path(__file__).resolve().parent
TONE_FILES = {
    "Casual": BASE_DIR.joinpath("streaming_ordering_chatbot/resources/casual.txt"),
    "GenZ": BASE_DIR.joinpath("streaming_ordering_chatbot/resources/genZ.txt"),
}

TONE_CHOICES = ["Default"] + list(TONE_FILES.keys())

TONE_TO_STYLE_MAPPING = {
    "Default": "default",
    "Casual": "casual", 
    "GenZ": "genz"
}

MODEL_CHOICES = {
    "GPT-4o": "gpt-4o",
    "GPT-4.1": "gpt-4.1"
}

async def fetch_brand_info():
    """Fetch brand information from the API."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f"{API_BASE_URL}/conversation-styles")
            if response.status_code == 200:
                data = response.json()
                return data.get("current_brand", "Restaurant")
            else:
                return "Restaurant"
    except Exception as e:
        st.warning(f"Could not connect to API: {e}")
        return "Restaurant"

def load_prompt(prompt_path):
    """Load prompt from a file with error handling."""
    try:
        with open(prompt_path, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        st.warning(f"Prompt file not found: {prompt_path}")
        return ""
    except Exception as e:
        st.error(f"Error loading prompt: {e}")
        return ""

async def generate_initial_greeting():
    """Generate an initial greeting using the brand personality system."""
    try:
        async with httpx.AsyncClient(timeout=10) as client:
            headers = {
                "brand-session-id": st.session_state.session_id,
                "request-id": str(uuid4()),
            }
            
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
                    "conversation_style": "default",
                    "deployment": "gpt-4o",
                },
            }
            
            response = await client.post(
                PREAMBLE_ENDPOINT,
                json=json_data,
                headers=headers
            )
            
            if response.status_code == 200:
                greeting = ""
                async for line in response.aiter_lines():
                    if line:
                        greeting += line
                return greeting.strip()
            else:
                return f"Welcome to {st.session_state.brand_name}! How can I help you today?"
                
    except Exception as e:
        st.warning(f"Could not generate dynamic greeting: {e}")
        return f"Welcome to {st.session_state.brand_name}! How can I help you today?"

async def main():
    st.set_page_config(
        page_title="Restaurant Ordering ChatBot with Evaluation"
    )
    
    # Initialize session ID FIRST - before evaluation system
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    
    # Initialize evaluation system AFTER session ID exists
    initialize_evaluation_system()
    
    # Explicitly initialize the evaluation session with the session ID
    if ("eval_system" in st.session_state and 
        st.session_state.eval_system and 
        hasattr(st.session_state.eval_system, 'init_session')):
        try:
            st.session_state.eval_system.init_session(st.session_state.session_id)
        except Exception as e:
            st.warning(f"Failed to initialize evaluation session: {e}")
    
    # Initialize brand name from API
    if "brand_name" not in st.session_state:
        st.session_state.brand_name = await fetch_brand_info()

    # Initialize chat history ONLY ONCE
    if "messages" not in st.session_state:
        initial_greeting = await generate_initial_greeting()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": initial_greeting,
                "filtered": False,
            }
        ]
        
    if "llm_order" not in st.session_state:
        st.session_state.llm_order = {"items": []}
        
    if "order" not in st.session_state:
        st.session_state.order = {"items": []}
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "GPT-4o"
    if "prompts" not in st.session_state:
        st.session_state.prompts = {k: load_prompt(v) for k, v in TONE_FILES.items()}
        st.session_state.selected_prompt = "Default"

    prompt = st.chat_input("Type your message here...")
    
    # Create layout
    chat_col, cart_col = st.columns([7, 3], gap="large")

    # Sidebar with controls and evaluation dashboard
    with st.sidebar:
        st.header("Options")
        
        tone_choice = st.selectbox("Choose a tone", options=TONE_CHOICES)
        if st.button("Apply Tone"):
            st.session_state.selected_prompt = tone_choice
            st.success(f"{tone_choice} tone applied!")

        model_choice = st.selectbox("Select Model", options=MODEL_CHOICES)
        if st.button("Select Model"):
            st.session_state.selected_model = model_choice
            st.success(f"Model updated to {model_choice}!")
            
        st.divider()
        
        # Evaluation dashboard
        create_evaluation_dashboard()

    # Cart column
    with cart_col:
        cart = st.empty()

    # Chat column
    with chat_col:
        # Display chat messages from history - NO re-initialization here
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # React to user input
        if prompt:
            # Display user message
            last_user = st.chat_message("user")
            last_user_content = last_user.empty()
            last_user_content.markdown(prompt)

            # Display assistant response
            last_asst = st.chat_message("assistant")
            chat_history = [
                message
                for message in st.session_state.messages
                if not message.get("filtered")
            ]
            
            with last_asst:
                # Content safety screening
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
                    
                # Handle content safety results
                if len(screening_result["failed_categories"]) > 0:
                    text = "I'm sorry, I can't process your request. Could you please try again?"
                    last_asst.markdown(text)
                    user_message = (
                        "<Redacted for content safety: "
                        + ", ".join(screening_result["failed_categories"])
                        + ">"
                    )
                    last_user_content.markdown(user_message)
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
                    
                    # Process based on intent
                    if screening_result["intent"] == "order":
                        # Handle order intent with evaluation
                        items_list = st.empty()
                        summary = st.empty()
                        
                        order = asyncio.create_task(
                            fetch_order_with_evaluation(
                                {"items": []},
                                cart,
                                items_list,
                                chat_history,
                                st.session_state,
                                st.session_state.eval_system,
                                prompt
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

                        # Wait for order processing
                        order_result = await order
                        
                        # Add order info to chat history
                        if order_result[1] is None:
                            order_info = "Failed to fetch order, item might not exist"
                        else:
                            order_info = order_result[0]
                        
                        chat_history.append({
                            "role": "assistant", 
                            "content": f"Order processed: {order_info}"
                        })
                        
                        # Generate summary response with evaluation
                        summary_response = await fetch_stream_with_evaluation(
                            SUMMARY_ENDPOINT,
                            summary,
                            json_data,
                            st.session_state.eval_system,
                            prompt,
                            "order_summary"
                        )
                        
                        assistant_message = {
                            "role": "assistant",
                            "content": summary_response,
                        }
                    else:
                        # Handle conversation intent with evaluation
                        conversation = st.empty()
                        response = await fetch_stream_with_evaluation(
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
                            st.session_state.eval_system,
                            prompt,
                            "conversation"
                        )
                        
                        assistant_message = {
                            "role": "assistant",
                            "content": response,
                        }
                    
                    # Add messages to session state
                    st.session_state.messages.append(user_message)
                    st.session_state.messages.append(assistant_message)

async def cleanup_resources():
    """Cleanup evaluation system resources on app shutdown."""
    if "eval_system" in st.session_state:
        if hasattr(st.session_state.eval_system, 'cleanup'):
            await st.session_state.eval_system.cleanup()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    finally:
        # Cleanup resources on shutdown
        if "eval_system" in st.session_state:
            try:
                asyncio.run(cleanup_resources())
            except Exception:
                pass  # Ignore cleanup errors