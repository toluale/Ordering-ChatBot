import asyncio
import logging
import os
import uuid
from typing import Optional

from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace
from opentelemetry.sdk.trace import SpanProcessor
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from promptflow.core import AzureOpenAIModelConfiguration
from promptflow.tracing import start_trace
from starlette.responses import StreamingResponse

from streaming_ordering_chatbot.api.content_safety import pre_process_check

from .flows.classification_flow import OrderIntentFlow
from .flows.conversation_flows import OrderAssistantFlow, PreambleFlow, SummaryFlow
from .flows.order_flow import OrderFlow
from .flows.schemas import LLMOrder
from .models import Message, OrderState, ScreenData, ScreeningResponse, LLMConfig


class SessionIdProcessor(SpanProcessor):
    def __init__(self, attributes: dict):
        self.attributes = attributes

    def on_start(self, span, parent_context):
        for attribute_name, attribute_value in self.attributes.items():
            span.set_attribute(attribute_name, attribute_value)

    def on_end(self, span):
        pass


# Promptflow tracing
start_trace()
exporter = AzureMonitorTraceExporter.from_connection_string(
    os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
)

trace_provider = trace.get_tracer_provider()
span_processor = BatchSpanProcessor(exporter, schedule_delay_millis=10000)
trace_provider.add_span_processor(span_processor)

logging.basicConfig(level=logging.INFO)


# Initialize the flow configurations
flow_configuration = AzureOpenAIModelConfiguration(
    azure_endpoint=os.environ["AZURE_ENDPOINT"],
    api_key=os.environ["AZURE_API_KEY"],
    api_version=os.environ["AZURE_API_VERSION"],
    azure_deployment=os.environ["AZURE_DEPLOYMENT_NAME"],
)

order_flow = OrderFlow(model_config=flow_configuration)
preamble_flow = PreambleFlow(model_config=flow_configuration)
summary_flow = SummaryFlow(model_config=flow_configuration)
intent_flow = OrderIntentFlow(model_config=flow_configuration)
assistant_flow = OrderAssistantFlow(model_config=flow_configuration)

app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_custom_attribute(request: Request, call_next):
    """
    Adds a custom attribute to the request headers.

    Args:
        request (Request): The incoming request object.
        call_next (Callable): The next middleware function to call.

    Returns:
        Response: The response object.

    """
    session_id = request.headers.get("contoso-session-id", str(uuid.uuid4()))
    request_id = request.headers.get("request-id", str(uuid.uuid4()))

    attributes = {"contoso-session-id": session_id, "request-id": request_id}

    trace.get_tracer_provider().add_span_processor(SessionIdProcessor(attributes))

    response = await call_next(request)
    return response


@app.post("/screen")
async def screen_message(data: ScreenData) -> ScreeningResponse:
    """
    Screens a message and returns the screening response.

    Args:
        data (ScreenData): The data containing the message to be screened.

    Returns:
        ScreeningResponse: Response containing the redacted message, failed categories, and intent.
    """
    chat_history = data.chat_history
    chat_history.append(
        Message.model_validate({"role": "user", "content": data.message})
    )
    pre_process_result = asyncio.create_task(pre_process_check(data.message))
    intent_result = asyncio.create_task(
        intent_flow(chat_history=chat_history, current_order=data.current_order)
    )
    results = await asyncio.gather(pre_process_result, intent_result)
    return ScreeningResponse(
        redacted_message=results[0][0],
        failed_categories=results[0][1],
        intent=results[1],
    )


@app.post("/order")
async def create_order(state: OrderState, config: LLMConfig):
    return StreamingResponse(
        order_flow(
            chat_history=state.chat_history,
            current_order=state.order,
            model_deployment=config.deployment,
        ),
        media_type="text/plain",
    )


@app.post("/preamble")
async def create_preamble(
    chat_history: list[Message],
    config: LLMConfig,
) -> StreamingResponse:
    return StreamingResponse(
        preamble_flow(
            chat_history=chat_history,
            personality=config.personality,
            model_deployment=config.deployment,
        ),
        media_type="text/plain",
    )


@app.post("/summary")
async def create_summary(
    chat_history: list[Message],
    config: LLMConfig,
) -> StreamingResponse:
    return StreamingResponse(
        summary_flow(
            chat_history=chat_history,
            personality=config.personality,
            model_deployment=config.deployment,
        ),
        media_type="text/plain",
    )


@app.post("/assistant")
async def assistant_response(
    chat_history: list[Message],
    current_order: LLMOrder,
    config: LLMConfig,
) -> StreamingResponse:
    return StreamingResponse(
        assistant_flow(
            chat_history=chat_history,
            current_order=current_order,
            personality=config.personality,
            model_deployment=config.deployment,
        ),
        media_type="text/plain",
    )
