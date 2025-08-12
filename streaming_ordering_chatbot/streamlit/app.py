import asyncio
import json
import time
import csv
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from typing import Dict, List, Any

import httpx
import streamlit as st
import pandas as pd
import os
import tiktoken

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
    "Gen Z": BASE_DIR.joinpath("resources/genZ.txt"),
}

TONE_CHOICES = ["Default"] + list(TONE_FILES.keys())

# Add this mapping to convert UI choices to API values
TONE_TO_STYLE_MAPPING = {
    "Default": "default",
    "Casual": "casual", 
    "Gen Z": "genz"
}

MODEL_CHOICES = {
    "GPT-4o": "gpt-4o",
    "GPT-4.1": "gpt-4.1"
}

# Evaluation metrics storage
EVALUATION_DATA_PATH = BASE_DIR.parent / "evaluation_results" / "streamlit_evaluation_metrics.csv"

class LatencyTracker:
    """Track latency and performance metrics for the chatbot."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.first_token_time = None
        self.token_count = 0
        
    def start(self):
        """Start timing the request."""
        self.start_time = time.perf_counter()
        self.first_token_time = None
        self.token_count = 0
        
    def first_token(self):
        """Mark when the first token is received."""
        if self.first_token_time is None:
            self.first_token_time = time.perf_counter()
            
    def add_token(self):
        """Increment token count."""
        self.token_count += 1
        
    def end(self):
        """End timing and calculate metrics."""
        self.end_time = time.perf_counter()
        
    def get_metrics(self) -> Dict[str, float]:
        """Get calculated timing metrics."""
        if not self.start_time or not self.end_time:
            return {}
            
        total_latency = self.end_time - self.start_time
        time_to_first_token = (self.first_token_time - self.start_time) if self.first_token_time else total_latency
        tokens_per_second = self.token_count / total_latency if total_latency > 0 else 0
        
        # Model latency is the time from first token to end (actual generation time)
        model_latency = (self.end_time - self.first_token_time) if self.first_token_time else total_latency
        
        return {
            "total_latency": total_latency,
            "model_latency": model_latency,
            "time_to_first_token": time_to_first_token,
            "tokens_per_second": tokens_per_second,
            "token_count": self.token_count
        }

# Initialize encoding for token counting
encoding = tiktoken.encoding_for_model("gpt-4o")

async def fetch_stream(url, container, json_data, tracker: LatencyTracker = None):
    """Enhanced fetch_stream with optional metrics tracking that doesn't interfere with streaming."""
    if tracker:
        tracker.start()
    
    async with httpx.AsyncClient(timeout=30) as client:
        headers = {
            "brand-session-id": st.session_state.session_id,
            "request-id": str(uuid4()),
        }
        async with client.stream(
            "POST", url, json=json_data, headers=headers
        ) as response:
            current_content = ""
            first_chunk = True
            
            async for line in response.aiter_lines():
                if line:
                    # Track first token timing (non-interfering)
                    if first_chunk and tracker:
                        tracker.first_token()
                        first_chunk = False
                    
                    # Original streaming logic - unchanged
                    if "<REDACTED" in line:
                        current_content = line
                    else:
                        current_content += line + "\n"
                    
                    # Post-process the content to ensure proper markdown formatting
                    container.markdown(current_content, unsafe_allow_html=True)
                    
                    # Count tokens after content processing (non-interfering)
                    if tracker:
                        try:
                            actual_tokens = len(encoding.encode(line))
                            for _ in range(actual_tokens):
                                tracker.add_token()
                        except:
                            # Fallback to simple counting if encoding fails
                            tracker.add_token()
    
    if tracker:
        tracker.end()
    
    return current_content

async def fetch_order(current_order, container, items_list, chat_history, state, tracker: LatencyTracker = None):
    """Enhanced fetch_order with optional metrics tracking."""
    if tracker:
        tracker.start()
    
    order_obj = ""
    first_line = True
    order_finished = False
    desc = []
    order = None
    first_chunk = True
    
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
                    # Track first token timing (non-interfering)
                    if first_chunk and tracker:
                        tracker.first_token()
                        first_chunk = False
                    
                    # Count tokens (non-interfering)
                    if tracker:
                        tracker.add_token()
                    
                    # Original order processing logic - unchanged
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
                                if tracker:
                                    tracker.end()
                                return "Error parsing order", None
                        else:
                            try:
                                order = json.loads(order_obj + "]}")
                            except json.JSONDecodeError:
                                continue
                        
                        # Safely extract descriptions (only if we have a valid order)
                        if order:
                            desc = []
                            order_items = order.get("order", [])
                            
                            for item in order_items:
                                description = None
                                
                                if "description" in item:
                                    description = item["description"]
                                elif "name" in item:
                                    name = item["name"]
                                    quantity = item.get("quantity", 1)
                                    description = f"{quantity}x {name}"
                                    
                                    if "size" in item:
                                        description += f" ({item['size']})"
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
    
    if tracker:
        tracker.end()
                            
    return ("- " + "\n - ".join(desc)) if desc else "No items", order

def save_evaluation_metrics(metrics: Dict[str, Any]):
    """Save evaluation metrics to CSV file."""
    EVALUATION_DATA_PATH.parent.mkdir(exist_ok=True)
    
    file_exists = EVALUATION_DATA_PATH.exists()
    metrics["timestamp"] = datetime.now().isoformat()
    
    with open(EVALUATION_DATA_PATH, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=metrics.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)

def load_evaluation_data() -> pd.DataFrame:
    """Load evaluation data from CSV file."""
    if EVALUATION_DATA_PATH.exists():
        return pd.read_csv(EVALUATION_DATA_PATH)
    return pd.DataFrame()

def display_evaluation_dashboard():
    """Display evaluation metrics dashboard in sidebar."""
    st.sidebar.header("Performance Metrics")
    
    df = load_evaluation_data()
    
    if df.empty:
        st.sidebar.info("No metrics data yet.")
        return
    
    # Basic stats
    st.sidebar.metric("Total Interactions", len(df))
    
    if 'total_latency' in df.columns:
        avg_latency = df['total_latency'].mean()
        st.sidebar.metric("Avg Total Latency", f"{avg_latency:.2f}s")
    
    if 'time_to_first_token' in df.columns:
        avg_ttft = df['time_to_first_token'].mean()
        st.sidebar.metric("Avg Time to First Token", f"{avg_ttft:.2f}s")
    
    if 'tokens_per_second' in df.columns:
        avg_tps = df['tokens_per_second'].mean()
        st.sidebar.metric("Avg Speed", f"{avg_tps:.1f} tok/s")

    if 'token_count' in df.columns:
        avg_tokens = df['token_count'].mean()
        st.sidebar.metric("Avg Token", f"{avg_tokens:.0f}")
    
    if 'model_latency' in df.columns:
        avg_model_latency = df['model_latency'].mean()
        st.sidebar.metric("Avg Model Latency", f"{avg_model_latency:.2f}s")
    
    # Show detailed dashboard button
    if st.sidebar.button("Show Detailed Analytics"):
        st.session_state.show_analytics = True

    # Clear data option
    if st.sidebar.button("Clear Metrics Data"):
        if EVALUATION_DATA_PATH.exists():
            EVALUATION_DATA_PATH.unlink()
            st.success("Metrics data cleared!")
            st.rerun()

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
    """Load prompt from a file."""
    with open(prompt_path, "r") as file:
        return file.read()

async def generate_initial_greeting():
    """Generate an initial greeting using the brand personality system."""
    try:
        async with httpx.AsyncClient(timeout=20) as client:
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
            
            current_style = TONE_TO_STYLE_MAPPING.get(
                getattr(st.session_state, 'selected_prompt', 'Default'), 
                "default"
            )
            
            json_data = {
                "chat_history": initial_chat_history,
                "config": {
                    "conversation_style": current_style,
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
    st.set_page_config(page_title="Ordering ChatBot")

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
    
    st.title(f"Ordering ChatBot")

    prompt = st.chat_input("")
    chat_col, cart_col = st.columns([7, 3], gap="large")

    with st.sidebar:
        tone_choice = st.selectbox("Choose a tone", options=TONE_CHOICES)

        if st.button("Apply Tone"):
            st.session_state.selected_prompt = tone_choice
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    await client.post(f"{API_BASE_URL}/clear-flow-cache")
                st.success(f"{tone_choice} tone applied successfully!")
            except:
                st.success(f"{tone_choice} tone applied! (Cache clear failed, but style should still work)")
                
            api_style = TONE_TO_STYLE_MAPPING.get(tone_choice, "default")
            st.info(f"Using conversation style: `{api_style}`")

        model_choice = st.selectbox("Select Model", options=MODEL_CHOICES)

        if st.button("Select Model"):
            st.session_state.selected_model = model_choice
            st.success(f"Model updated to {model_choice}!")

        # Add evaluation dashboard
        st.divider()
        display_evaluation_dashboard()

    with cart_col:
        cart = st.empty()

    with chat_col:
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

            # Display assistant response in chat message container
            last_asst = st.chat_message("assistant")
            chat_history = [
                message
                for message in st.session_state.messages
                if not message.get("filtered")
            ]
            
            # Initialize evaluation metrics
            evaluation_metrics = {
                "user_input": prompt,
                "model": st.session_state.selected_model,
                "conversation_style": TONE_TO_STYLE_MAPPING.get(st.session_state.selected_prompt, "default"),
                "session_id": st.session_state.session_id,
            }
            
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
                    
                    # Update metrics for filtered content
                    evaluation_metrics.update({
                        "intent": "filtered",
                        "content_filtered": True,
                        "total_latency": 0,
                        "model_latency": 0,
                        "time_to_first_token": 0,
                        "tokens_per_second": 0,
                        "token_count": 0
                    })
                    
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
                    
                    # Create tracker for metrics
                    tracker = LatencyTracker()
                    
                    if screening_result["intent"] == "order":
                        items_list = st.empty()
                        summary = st.empty()
                        
                        # Process order with metrics tracking
                        order = asyncio.create_task(
                            fetch_order(
                                {"items": []},
                                cart,
                                items_list,
                                chat_history,
                                st.session_state,
                                tracker  # Add tracker here
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

                        order_result = await order
                        
                        if order_result[1] is None:
                            order_info = "Failed to fetch order, item might have not existed"
                        else:
                            order_info = order_result[0]
                        
                        chat_history.append({
                            "role": "assistant", 
                            "content": f"Order processed: {order_info}"
                        })
                        
                        # Generate summary response with separate tracker
                        summary_tracker = LatencyTracker()
                        summary_response = await fetch_stream(
                            SUMMARY_ENDPOINT,
                            summary,
                            json_data,
                            summary_tracker  # Add tracker here
                        )
                        
                        # Combine metrics
                        order_metrics = tracker.get_metrics()
                        summary_metrics = summary_tracker.get_metrics()
                        
                        evaluation_metrics.update({
                            "intent": "order",
                            "content_filtered": False,
                            "total_latency": summary_metrics.get("total_latency", 0),
                            "model_latency": summary_metrics.get("model_latency", 0),
                            "time_to_first_token": order_metrics.get("time_to_first_token", 0),
                            "tokens_per_second": (summary_metrics.get("token_count", 0)) / 
                                               (summary_metrics.get("total_latency", 0)) 
                                               if (summary_metrics.get("total_latency", 0)) > 0 else 0,
                            "token_count": summary_metrics.get("token_count", 0)
                        })
                        
                        assistant_message = {
                            "role": "assistant",
                            "content": summary_response,
                        }
                    else:
                        conversation = st.empty()
                        
                        # Process conversation with metrics tracking
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
                            tracker  # Add tracker here
                        )

                        # Update metrics
                        metrics = tracker.get_metrics()
                        evaluation_metrics.update({
                            "intent": "conversation",
                            "content_filtered": False,
                            **metrics
                        })

                        assistant_message = {
                            "role": "assistant",
                            "content": response,
                        }
                    
                    st.session_state.messages.append(user_message)
                    st.session_state.messages.append(assistant_message)
                
                # Save evaluation metrics
                save_evaluation_metrics(evaluation_metrics)
                
                # Show real-time metrics in sidebar
                if evaluation_metrics.get("total_latency"):
                    st.sidebar.success(f"Response: {evaluation_metrics['total_latency']:.2f}s")
                    if evaluation_metrics.get("model_latency"):
                        st.sidebar.info(f"Model latency: {evaluation_metrics['model_latency']:.2f}s")
                    if evaluation_metrics.get("time_to_first_token"):
                        st.sidebar.info(f"First token: {evaluation_metrics['time_to_first_token']:.2f}s")
                    if evaluation_metrics.get("token_count"):
                        st.sidebar.info(f"Tokens: {evaluation_metrics['token_count']}")
                    if evaluation_metrics.get("tokens_per_second"):
                        st.sidebar.info(f"Speed: {evaluation_metrics['tokens_per_second']:.1f} tok/s")

if __name__ == "__main__":
    asyncio.run(main())