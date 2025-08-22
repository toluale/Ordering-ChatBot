import asyncio
import json
import time
import csv
from datetime import datetime
from pathlib import Path
from uuid import uuid4
from typing import Dict, List, Any, Optional

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
        model_latency = (self.first_token_time - self.start_time) if self.first_token_time else total_latency
        generation_latency = (self.end_time - self.first_token_time) if self.first_token_time else total_latency
        
        # Use generation_latency for pure generation speed (excluding network/queue time)
        tokens_per_second = self.token_count / max(generation_latency, 0.01) if generation_latency > 0 else 0

        return {
            "total_latency": total_latency,
            "model_latency": model_latency,  # Time to first token (network + queue + model startup)
            "generation_latency": generation_latency,  # Pure generation time
            "tokens_per_second": tokens_per_second,  # Pure generation speed
            "token_count": self.token_count
        }

encoding = tiktoken.encoding_for_model("gpt-4o")
async def fetch_stream_with_metrics(url, container, json_data, tracker: LatencyTracker):
    """Enhanced fetch_stream that tracks timing metrics."""
    tracker.start()
    current_content = ""  # Initialize here to ensure it's always available
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            headers = {
                "brand-session-id": st.session_state.session_id,
                "request-id": str(uuid4()),
            }
            async with client.stream(
                "POST", url, json=json_data, headers=headers
            ) as response:
                first_chunk = True
                
                async for line in response.aiter_lines():
                    if line:
                        if first_chunk:
                            tracker.first_token()
                            first_chunk = False
                        
                        if "<REDACTED" in line:
                            # All previous content is sent with redacted information
                            current_content = line
                        else:
                            current_content += line + "\n"
                        
                        # Post-process the content to ensure proper markdown formatting
                        container.markdown(current_content, unsafe_allow_html=True)
                        
                        # Count tokens AFTER content processing to avoid interference
                        actual_tokens = len(encoding.encode(line))
                        for _ in range(actual_tokens):
                            tracker.add_token()
                            
    except httpx.ReadTimeout:
        st.error("Request timed out. The server took too long to respond.")
        current_content = "I apologize, but the request timed out. Please try again."
        container.markdown(current_content)
    except httpx.RequestError as e:
        st.error(f"Connection error: {str(e)}")
        current_content = "I'm having trouble connecting to the server. Please check your connection and try again."
        container.markdown(current_content)
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        current_content = "An unexpected error occurred. Please try again."
        container.markdown(current_content)
    
    tracker.end()
    return current_content

def _format_order_item(item: Dict[str, Any]) -> Optional[str]:
    """Format a single order item for display."""
    if "description" in item:
        return item["description"]
    elif "name" in item:
        name = item["name"]
        quantity = item.get("quantity", 1)
        description = f"{quantity}x {name}"
        
        if "size" in item:
            description += f" ({item['size']})"
        return description
    else:
        return str(item)

def _update_order_display(order: Dict[str, Any], container, items_list) -> List[str]:
    """Update the order display containers."""
    desc = []
    order_items = order.get("order", [])
    
    for item in order_items:
        description = _format_order_item(item)
        if description:
            desc.append(description)
    
    container.markdown(order)
    if desc:
        items_list.markdown("- " + "\n - ".join(desc))
    else:
        items_list.markdown("- No items in order")
    
    return desc

async def fetch_order_with_metrics(current_order, container, items_list, chat_history, state, tracker: LatencyTracker):
    """Enhanced fetch_order that tracks timing metrics."""
    tracker.start()
    
    order_obj = ""
    first_line = True
    order_finished = False
    desc = []
    order = None
    first_chunk = True
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:  # Increased timeout
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
                        "conversation_style": TONE_TO_STYLE_MAPPING.get(
                            st.session_state.selected_prompt, "default"
                        ),
                        "deployment": MODEL_CHOICES[st.session_state.selected_model]
                    },
                },
                headers=headers,
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        if first_chunk:
                            tracker.first_token()
                            first_chunk = False
                        
                        tracker.add_token()
                        
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
                                    tracker.end()
                                    return "Error parsing order", None
                            else:
                                try:
                                    order = json.loads(order_obj + "]}")
                                except json.JSONDecodeError:
                                    continue
                            
                            if order:
                                desc = _update_order_display(order, container, items_list)
    except httpx.ReadTimeout:
        st.error("Order processing timed out. The server took too long to respond.")
        container.markdown("Order processing timed out. Please try again.")
        items_list.markdown("- Order processing failed due to timeout")
        tracker.end()
        return "Order processing timed out", None
    except httpx.RequestError as e:
        st.error(f"Connection error during order processing: {str(e)}")
        container.markdown("Connection error during order processing.")
        items_list.markdown("- Order processing failed due to connection error")
        tracker.end()
        return "Order processing failed", None
    except Exception as e:
        st.error(f"Unexpected error during order processing: {str(e)}")
        container.markdown("Unexpected error during order processing.")
        items_list.markdown("- Order processing failed due to unexpected error")
        tracker.end()
        return "Order processing failed", None
                            
    tracker.end()
    return ("- " + "\n - ".join(desc)) if desc else "No items", order

def save_evaluation_metrics(metrics: Dict[str, Any]):
    """Save evaluation metrics to CSV file."""
    # evaluation_results directory
    EVALUATION_DATA_PATH.parent.mkdir(exist_ok=True)
    
    # Check if file exists to determine if we need headers
    file_exists = EVALUATION_DATA_PATH.exists()
    
    # Add timestamp
    metrics["timestamp"] = datetime.now().isoformat()
    
    # Write to CSV
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
    """Display evaluation metrics dashboard."""
    st.sidebar.header("Latency Tracker Dashboard")
    
    df = load_evaluation_data()
    
    if df.empty:
        st.sidebar.info("No evaluation data available yet.")
        return
    
    # Basic stats
    st.sidebar.metric("Total Interactions", len(df))
    
    if 'total_latency' in df.columns:
        avg_latency = df['total_latency'].mean()
        st.sidebar.metric("Avg Total Latency", f"{avg_latency:.2f}s")
    
    if 'token_count' in df.columns:
        avg_tokens = df['token_count'].mean()
        st.sidebar.metric("Avg Token Count", f"{avg_tokens:.0f}")
    
    if 'model_latency' in df.columns:
        avg_model_latency = df['model_latency'].mean()
        st.sidebar.metric("Avg Model Latency", f"{avg_model_latency:.2f}s")
    
    # Show detailed dashboard button
    if st.sidebar.button("Show Detailed Analytics"):
        st.session_state.show_analytics = True

def _display_recent_interactions(df: pd.DataFrame):
    """Display recent interactions table."""
    st.subheader("Recent Interactions")
    recent_df = df.tail(10)
    if not recent_df.empty:
        display_cols = ['timestamp', 'intent', 'model', 'conversation_style', 
                       'total_latency', 'model_latency', 
                       'tokens_per_second', 'token_count']
        display_cols = [col for col in display_cols if col in recent_df.columns]
        st.dataframe(recent_df[display_cols].sort_values('timestamp', ascending=False))

def display_detailed_analytics():
    """Display detailed analytics in the main area."""
    st.header("Chatbot Performance Analytics")
    
    df = load_evaluation_data()
    
    if df.empty:
        st.info("No evaluation data available yet. Start chatting to collect metrics!")
        return
    
    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    _display_recent_interactions(df)
    
    # Export data
    if st.button("Export Data as CSV"):
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"chatbot_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

async def fetch_brand_info():
    """Fetch brand information from the API."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:  # Increased timeout
            response = await client.get(f"{API_BASE_URL}/conversation-styles")
            if response.status_code == 200:
                data = response.json()
                return data.get("current_brand", "Restaurant")
            else:
                return "Restaurant"
    except httpx.ReadTimeout:
        st.warning("Could not fetch brand info (timeout) - using default")
        return "Restaurant"
    except httpx.RequestError as e:
        st.warning(f"Could not connect to API: {e}")
        return "Restaurant"
    except Exception as e:
        st.warning(f"Unexpected error fetching brand info: {e}")
        return "Restaurant"

def load_prompt(prompt_path):
    """Load prompt from a file."""
    with open(prompt_path, "r") as file:
        return file.read()

async def generate_initial_greeting():
    """Generate an initial greeting using the brand personality system."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:  # Increased timeout
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
                    "deployment": "GPT-4o",
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
                
    except httpx.ReadTimeout:
        st.warning("Could not generate greeting (timeout) - using default")
        return f"Welcome to {st.session_state.brand_name}! How can I help you today?"
    except httpx.RequestError as e:
        st.warning(f"Could not generate dynamic greeting: {e}")
        return f"Welcome to {st.session_state.brand_name}! How can I help you today?"
    except Exception as e:
        st.warning(f"Unexpected error generating greeting: {e}")
        return f"Welcome to {st.session_state.brand_name}! How can I help you today?"

def _initialize_session_state():
    """Initialize all session state variables."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "llm_order" not in st.session_state:
        st.session_state.llm_order = {"items": []}
    if "order" not in st.session_state:
        st.session_state.order = {"items": []}
    if "selected_model" not in st.session_state:
        st.session_state.selected_model = "GPT-4o"
    if "prompts" not in st.session_state:
        st.session_state.prompts = {k: load_prompt(v) for k, v in TONE_FILES.items()}
        st.session_state.selected_prompt = "Default"
    if "show_analytics" not in st.session_state:
        st.session_state.show_analytics = False
    if "evaluation_mode" not in st.session_state:
        st.session_state.evaluation_mode = True
    if "clear_cache_requested" not in st.session_state:
        st.session_state.clear_cache_requested = False

async def _setup_initial_greeting():
    """Setup initial greeting if not already done."""
    if "brand_name" not in st.session_state:
        st.session_state.brand_name = await fetch_brand_info()
    if "messages" not in st.session_state:
        # Ensure session_id is initialized before generating greeting
        if "session_id" not in st.session_state:
            st.session_state.session_id = str(uuid4())
        initial_greeting = await generate_initial_greeting()
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": initial_greeting,
                "filtered": False,
            }
        ]
async def _handle_cache_clearing():
    """Handle cache clearing if requested."""
    if getattr(st.session_state, 'clear_cache_requested', False):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(f"{API_BASE_URL}/clear-flow-cache")
            st.session_state.clear_cache_requested = False
            return True
        except Exception as e:
            st.session_state.clear_cache_requested = False
            return False
    return True

def _render_sidebar_controls():
    """Render sidebar controls and return user selections."""
    with st.sidebar:
        tone_choice = st.selectbox("Choose a tone", options=TONE_CHOICES)

        if st.button("Apply Tone"):
            st.session_state.selected_prompt = tone_choice
            # Clear backend cache to ensure new style is applied
            st.session_state.clear_cache_requested = True
            st.success(f"{tone_choice} tone applied successfully!")
                            
            # Show which API style value is being used
            api_style = TONE_TO_STYLE_MAPPING.get(tone_choice, "default")
            st.info(f"Using conversation style: `{api_style}`")

        model_choice = st.selectbox("Select Model", options=MODEL_CHOICES)
        if st.button("Select Model"):
            st.session_state.selected_model = model_choice
            st.success(f"Model updated to {model_choice}!")
        
        # Evaluation controls
        st.divider()
        display_evaluation_dashboard()
        
        # Clear data option
        if st.button("Clear Evaluation Data"):
            if EVALUATION_DATA_PATH.exists():
                EVALUATION_DATA_PATH.unlink()
                st.success("Evaluation data cleared!")
                st.rerun()

async def _handle_content_screening(prompt: str, chat_history: List[Dict], evaluation_metrics: Dict) -> tuple:
    """Handle content screening and return results."""
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:  # Increased timeout
            headers = {
                "brand-session-id": st.session_state.session_id,
                "request-id": str(uuid4()),
            }
            data = {
                "message": prompt,
                "chat_history": chat_history,
                "current_order": st.session_state.llm_order,
            }
            res = await client.post(SCREENING_ENDPOINT, json=data, headers=headers)
            screening_result = res.json()
    except httpx.ReadTimeout:
        st.error("Content screening timed out.")
        # Return a safe default response for timeout
        screening_result = {
            "failed_categories": [],
            "redacted_message": prompt,
            "intent": "conversation"
        }
    except httpx.RequestError as e:
        st.error(f"Connection error during screening: {str(e)}")
        # Return a safe default response for connection error
        screening_result = {
            "failed_categories": [],
            "redacted_message": prompt,
            "intent": "conversation"
        }
    except Exception as e:
        st.error(f"Unexpected error during screening: {str(e)}")
        # Return a safe default response for unexpected error
        screening_result = {
            "failed_categories": [],
            "redacted_message": prompt,
            "intent": "conversation"
        }
    
    if len(screening_result["failed_categories"]) > 0:
        evaluation_metrics.update({
            "intent": "filtered",
            "content_filtered": True,
            "filter_categories": ", ".join(screening_result["failed_categories"]),
            "total_latency": 0,
            "model_latency": 0,
            "tokens_per_second": 0,
            "token_count": 0
        })
        return screening_result, True, evaluation_metrics
    else:
        evaluation_metrics.update({
            "intent": screening_result["intent"],
            "content_filtered": False,
        })
        return screening_result, False, evaluation_metrics

async def _process_order_intent(chat_history: List[Dict], tracker: LatencyTracker, cart, items_list):
    """Process order intent and return results."""
    order_result = await fetch_order_with_metrics(
        {"items": []}, cart, items_list, chat_history, st.session_state, tracker
    )
    
    json_data = {
        "chat_history": chat_history,
        "config": {
            "conversation_style": TONE_TO_STYLE_MAPPING.get(st.session_state.selected_prompt, "default"),
            "deployment": MODEL_CHOICES[st.session_state.selected_model],
        },
    }

    # Add order info to chat history
    order_info = "Failed to fetch order, item might have not existed" if order_result[1] is None else order_result[0]
    chat_history.append({"role": "assistant", "content": f"Order processed: {order_info}"})
    
    # Generate summary response
    summary = st.empty()
    summary_tracker = LatencyTracker()
    summary_response = await fetch_stream_with_metrics(SUMMARY_ENDPOINT, summary, json_data, summary_tracker)
    
    
    return summary_response, summary_tracker.get_metrics() #combined_metrics

async def _process_conversation_intent(chat_history: List[Dict], tracker: LatencyTracker):
    """Process conversation intent and return results."""
    conversation = st.empty()
    response = await fetch_stream_with_metrics(
        CONVERSATION_ENDPOINT,
        conversation,
        {
            "chat_history": chat_history,
            "current_order": st.session_state.llm_order,
            "config": {
                "conversation_style": TONE_TO_STYLE_MAPPING.get(st.session_state.selected_prompt, "default"),
                "deployment": MODEL_CHOICES[st.session_state.selected_model],
            },
        },
        tracker
    )

    return response, tracker.get_metrics()

async def main():
    st.set_page_config(page_title="Ordering ChatBot")
    
    # Initialize session state first
    _initialize_session_state()
    
    await _handle_cache_clearing()
    
    # Then setup initial greeting (which may need session_id)
    await _setup_initial_greeting()

    # Header
    #st.title(f"{st.session_state.brand_name} Ordering ChatBot")
    st.title(f"Ordering ChatBot")
    
    # Show analytics view if requested
    if st.session_state.show_analytics:
        if st.button(" Back to Chat"):
            st.session_state.show_analytics = False
            st.rerun()
        display_detailed_analytics()
        return

    prompt = st.chat_input("")
    
    # Main layout
    chat_col, cart_col = st.columns([7, 3], gap="large")

    # Sidebar controls
    _render_sidebar_controls()

    # Cart display
    with cart_col:
        cart = st.empty()

    # Chat interface
    with chat_col:
        # Display chat messages from history
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
            chat_history = [msg for msg in st.session_state.messages if not msg.get("filtered")]
            
            # Initialize metrics tracking
            evaluation_metrics = {
                "user_input": prompt,
                "model": st.session_state.selected_model,
                "conversation_style": TONE_TO_STYLE_MAPPING.get(st.session_state.selected_prompt, "default"),
                "session_id": st.session_state.session_id,
            }
            
            with last_asst:
                # Content screening
                screening_result, is_filtered, evaluation_metrics = await _handle_content_screening(
                    prompt, chat_history, evaluation_metrics
                )
                
                if is_filtered:
                    text = "I'm sorry, I can't process your request. Could you please try again?"
                    last_asst.markdown(text)
                    user_message = f"<Redacted for content safety: {', '.join(screening_result['failed_categories'])}>"
                    last_user_content.markdown(user_message)
                    
                    st.session_state.messages.extend([
                        {"role": "user", "content": user_message, "filtered": True},
                        {"role": "assistant", "content": text, "filtered": True}
                    ])
                else:
                    last_user_content.markdown(screening_result["redacted_message"])
                    user_message = {
                        "role": "user",
                        "content": screening_result["redacted_message"],
                        "filtered": False,
                    }
                    chat_history.append(user_message)
                    
                    # Create latency tracker
                    tracker = LatencyTracker()
                    
                    if screening_result["intent"] == "order":
                        items_list = st.empty()
                        response, combined_metrics = await _process_order_intent(
                            chat_history, tracker, cart, items_list
                        )
                        assistant_message = {"role": "assistant", "content": response}
                    else:
                        response, combined_metrics = await _process_conversation_intent(chat_history, tracker)
                        assistant_message = {"role": "assistant", "content": response}
                    
                    # Update evaluation metrics with timing data
                    evaluation_metrics.update(combined_metrics)
                    
                    st.session_state.messages.extend([user_message, assistant_message])
            
            # Save evaluation metrics
            save_evaluation_metrics(evaluation_metrics)
            
            # Show real-time metrics
            if evaluation_metrics.get("total_latency"):
                st.sidebar.success(f"Response time: {evaluation_metrics['total_latency']:.2f}s")
                if evaluation_metrics.get("model_latency"):
                    st.sidebar.info(f"Model latency: {evaluation_metrics['model_latency']:.2f}s")
                if evaluation_metrics.get("token_count"):
                    st.sidebar.info(f"Tokens: {evaluation_metrics['token_count']}")
                if evaluation_metrics.get("tokens_per_second"):
                    st.sidebar.info(f"Speed: {evaluation_metrics['tokens_per_second']:.1f} tokens/s")

if __name__ == "__main__":
    asyncio.run(main())
