import asyncio
import json
from pathlib import Path
from uuid import uuid4

import httpx

import streamlit as st

ORDER_ENDPOINT = "http://localhost:8000/order"
PREAMBLE_ENDPOINT = "http://localhost:8000/preamble"
SUMMARY_ENDPOINT = "http://localhost:8000/summary"
SCREENING_ENDPOINT = "http://localhost:8000/screen"
CONVERSATION_ENDPOINT = "http://localhost:8000/assistant"

BASE_DIR = Path(__file__).resolve().parent.parent
PREAMBLE_PROMPT_PATH = BASE_DIR.joinpath("api/flows/prompts/preamble.prompty")
SUMMARY_PROMPT_PATH = BASE_DIR.joinpath("api/flows/prompts/summary.prompty")

DEFAULT_PREAMBLE_PATH = BASE_DIR.joinpath("resources/default_preamble.txt")
DEFAULT_SUMMARY_PATH = BASE_DIR.joinpath("resources/default_summary.txt")

TONE_FILES = {
    "Casual": BASE_DIR.joinpath("resources/casual.txt"),
    "GenZ": BASE_DIR.joinpath("resources/genZ.txt"),
}


TONE_CHOICES = ["Default"] + list(TONE_FILES.keys())


MODEL_CHOICES = {
    "GPT-4o": "gpt-4o",
    "GPT-4o-mini": "gpt-4o-mini",
    "GPT-4-turbo": "gpt-4-turbo",
    "GPT-3.5-turbo": "gpt-35-turbo",
}


async def fetch_stream(url, container, json_data):
    async with httpx.AsyncClient(timeout=30) as client:
        headers = {
            "contoso-session-id": st.session_state.session_id,
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
                        current_content += line
                    container.markdown(current_content)
    return current_content


async def fetch_order(current_order, container, items_list, chat_history, state):
    order_obj = ""
    first_line = True
    order_finished = False
    async with httpx.AsyncClient(timeout=30) as client:
        headers = {
            "contoso-session-id": st.session_state.session_id,
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
                        state.llm_order = json.loads(line)["LLMOrder"]
                    else:
                        order_obj += line
                        if line == "]}":
                            order_finished = True
                            order = json.loads(order_obj)
                        else:
                            try:
                                order = json.loads(
                                    order_obj + "]}"
                                )  # Attempt to decode the JSON
                            except json.JSONDecodeError:
                                return None
                        desc = []
                        for item in order["order"]:
                            if description := item.pop("description"):
                                desc.append(description)
                        container.markdown(order)
                        items_list.markdown("- " + "\n - ".join(desc))
    return "- " + "\n - ".join(desc), order


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


async def main():

    # st.set_page_config(layout="wide")
    # Initialize chat history
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid4())
    if "llm_order" not in st.session_state:
        st.session_state.llm_order = {"items": []}
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Welcome to Contoso Burger's ordering chatbot! How can I help you today?",
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
                        "contoso-session-id": st.session_state.session_id,
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
                        preamble = st.empty()
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
                                "personality": st.session_state.prompts.get(
                                    st.session_state.selected_prompt
                                ),
                                "deployment": MODEL_CHOICES[
                                    st.session_state.selected_model
                                ],
                            },
                        }

                        preamble = asyncio.create_task(
                            fetch_stream(
                                PREAMBLE_ENDPOINT,
                                preamble,
                                json_data,
                            )
                        )
                        # Execute ordering and preamble concurrently
                        results = await asyncio.gather(preamble, order)

                        # Add partial assistant response to chat history for summary generation
                        if results[1] is None:
                            assistant_response = (
                                results[0]
                                + "\n"
                                + "Failed to fetch order, item might have not existed"
                            )
                        else:
                            assistant_response = results[0] + "\n" + results[1][0]
                        assistant_message = {
                            "role": "assistant",
                            "content": assistant_response,
                        }
                        chat_history.append(assistant_message)
                        summary = await fetch_stream(
                            SUMMARY_ENDPOINT,
                            summary,
                            json_data,
                        )
                        assistant_message["content"] += "\n\n" + summary
                    else:
                        conversation = st.empty()
                        response = await fetch_stream(
                            CONVERSATION_ENDPOINT,
                            conversation,
                            {
                                "chat_history": chat_history,
                                "current_order": st.session_state.llm_order,
                                "config": {
                                    "personality": st.session_state.prompts.get(
                                        st.session_state.selected_prompt
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
