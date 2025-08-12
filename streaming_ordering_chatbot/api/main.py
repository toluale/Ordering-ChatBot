import asyncio
import logging
import os
import uuid
from typing import Optional

from dotenv import load_dotenv

from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from starlette.responses import StreamingResponse

from streaming_ordering_chatbot.api.content_safety import pre_process_check, wrap_content_safety

from .flows.classification_flow_SK import OrderIntentFlowSK
from .flows.conversation_flows_SK import OrderAssistantFlowSK, PreambleFlowSK, SummaryFlowSK
from .flows.conversation_style import ConversationStyle
from .flows.order_flow_SK import OrderFlowSK
from .flows.schemas_generalized import LLMOrder
from .models import Message, OrderState, ScreenData, ScreeningResponse, LLMConfig


# Telemetry setup
logging.basicConfig(level=logging.INFO)

# Initialize telemetry
try:
    exporter = AzureMonitorTraceExporter.from_connection_string(
        os.environ["APPLICATIONINSIGHTS_CONNECTION_STRING"]
    )
    
    trace_provider = TracerProvider()
    trace.set_tracer_provider(trace_provider)
    
    span_processor = BatchSpanProcessor(exporter, schedule_delay_millis=10000)
    trace_provider.add_span_processor(span_processor)
    
    logging.info("Application Insights telemetry initialized successfully")
except Exception as e:
    logging.warning(f"Failed to initialize telemetry: {e}")
    # Continue without telemetry if setup fails
load_dotenv()

def get_required_env_var(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"{name} environment variable is not set. Please set it in your .env file."
        )
    return value

# Get Azure OpenAI configuration from environment
ENDPOINT = get_required_env_var("AZURE_OPENAI_ENDPOINT")
API_KEY = get_required_env_var("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = get_required_env_var("AZURE_OPENAI_DEPLOYMENT_NAME")

# Get brand name and default conversation style from environment
BRAND_NAME = get_required_env_var("BRAND_NAME")
CONVERSATION_STYLE = os.getenv("CONVERSATION_STYLE", "default")

# Initialize the Semantic Kernel flows with default conversation style
order_flow = OrderFlowSK(
    endpoint=ENDPOINT,
    api_key=API_KEY,
    deployment_name=DEPLOYMENT_NAME,
    brand_name=BRAND_NAME
)

_flow_cache = {}
# Create a factory function to get flows with specific conversation styles
def get_conversation_flow(flow_class, conversation_style: Optional[str] = None):
    """Get a conversation flow instance with the specified conversation style."""
    style = conversation_style or CONVERSATION_STYLE
    
    # Validate conversation style
    valid_styles = [style.value for style in ConversationStyle]
    if style not in valid_styles:
        logging.warning(f"Invalid conversation style '{style}', falling back to default")
        style = CONVERSATION_STYLE
    
    cache_key = f"{flow_class.__name__}_{style}"
    
    # Check if we have this specific flow+style combination cached
    if cache_key in _flow_cache:
        logging.debug(f"Using cached flow: {cache_key}")
        return _flow_cache[cache_key]
    
    # Create new flow with the requested conversation style
    flow = flow_class(
        ENDPOINT=ENDPOINT,
        API_KEY=API_KEY,
        DEPLOYMENT_NAME=DEPLOYMENT_NAME,
        BRAND_NAME=BRAND_NAME,
        CONVERSATION_STYLE=style
    )
    
    # Cache the flow for future use
    _flow_cache[cache_key] = flow
    logging.info(f"Created and cached new flow: {cache_key}")
    
    return flow

# Initialize intent classification flow (no conversation style needed)
intent_flow = OrderIntentFlowSK(
    ENDPOINT=ENDPOINT,
    API_KEY=API_KEY,
    DEPLOYMENT_NAME=DEPLOYMENT_NAME,
    BRAND_NAME=BRAND_NAME
)

# Initialize default flows (can be overridden per request)
preamble_flow = get_conversation_flow(PreambleFlowSK, CONVERSATION_STYLE)
summary_flow = get_conversation_flow(SummaryFlowSK, CONVERSATION_STYLE)
assistant_flow = get_conversation_flow(OrderAssistantFlowSK, CONVERSATION_STYLE)

app = FastAPI()

origins = ["http://localhost:3000", "http://localhost:8501"]

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
    Adds a custom attribute to the request headers and spans.

    Args:
        request (Request): The incoming request object.
        call_next (Callable): The next middleware function to call.

    Returns:
        Response: The response object.
    """
    session_id = request.headers.get("brand-session-id", str(uuid.uuid4()))
    request_id = request.headers.get("request-id", str(uuid.uuid4()))

    # Add custom attributes to current span if available
    try:
        tracer = trace.get_tracer(__name__)
        with tracer.start_as_current_span("http_request") as span:
            span.set_attribute("brand-session-id", session_id)
            span.set_attribute("request-id", request_id)
            response = await call_next(request)
        return response
    except Exception:
        # If tracing fails, continue without it
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
    
    # Convert LLMOrder to dict for intent classification
    current_order_dict = data.current_order.model_dump() if isinstance(data.current_order, LLMOrder) else data.current_order
    intent_result = asyncio.create_task(
        intent_flow(chat_history=chat_history, current_order=current_order_dict)
    )
    results = await asyncio.gather(pre_process_result, intent_result)
    return ScreeningResponse(
        redacted_message=results[0][0],
        failed_categories=results[0][1],
        intent=results[1],
    )


@app.post("/order")
async def create_order(state: OrderState, config: LLMConfig):
    """
    Create order using the SK-based order flow.
    
    Args:
        state: Order state containing chat history and current order
        config: LLM configuration including conversation style
    
    Returns:
        StreamingResponse: Streaming order response
    """
    # Convert LLMOrder to dict for SK flow
    current_order_dict = state.order.model_dump() if isinstance(state.order, LLMOrder) else state.order
    
    return StreamingResponse(
        order_flow(
            chat_history=state.chat_history,
            current_order=current_order_dict,
            model_deployment=config.deployment,
        ),
        media_type="text/plain",
    )


@app.post("/preamble")
async def create_preamble(
    chat_history: list[Message],
    config: LLMConfig,
) -> StreamingResponse:
    """
    Create preamble/greeting using the SK-based preamble flow.
    
    Args:
        chat_history: Chat conversation history
        config: LLM configuration including conversation style
    
    Returns:
        StreamingResponse: Streaming preamble response
    """
    # Get flow instance with the requested conversation style
    flow = get_conversation_flow(PreambleFlowSK, config.conversation_style)
    
    return StreamingResponse(
        flow(
            chat_history=chat_history,
            model_deployment=config.deployment,
        ),
        media_type="text/plain",
    )


@app.post("/summary")
async def create_summary(
    chat_history: list[Message],
    config: LLMConfig,
) -> StreamingResponse:
    """
    Create order summary using the SK-based summary flow.
    
    Args:
        chat_history: Chat conversation history
        config: LLM configuration including conversation style
    
    Returns:
        StreamingResponse: Streaming summary response
    """
    # Get flow instance with the requested conversation style
    flow = get_conversation_flow(SummaryFlowSK, config.conversation_style)
    
    return StreamingResponse(
        flow(
            chat_history=chat_history,
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
    """
    Generate assistant response using the SK-based assistant flow.
    """
    logging.info(f"Assistant request with conversation style: {config.conversation_style}")
    logging.info(f"Current cache keys: {list(_flow_cache.keys())}")
    
    # Convert LLMOrder to dict for SK flow
    current_order_dict = current_order.model_dump() if isinstance(current_order, LLMOrder) else current_order
    
    # Get flow instance with the requested conversation style
    flow = get_conversation_flow(OrderAssistantFlowSK, config.conversation_style)
    
    return StreamingResponse(
        flow(
            chat_history=chat_history,
            current_order=current_order_dict,
            model_deployment=config.deployment,
        ),
        media_type="text/plain",
    )

@app.get("/conversation-styles")
async def list_conversation_styles():
    """
    List available conversation styles.
    
    Returns:
        dict: Dictionary of available conversation styles with descriptions
    """
    return {
        "available_styles": {
            ConversationStyle.DEFAULT.value: "Standard brand personality only",
            ConversationStyle.CASUAL.value: "Casual, friendly, buddy-like conversation",
            ConversationStyle.GENZ.value: "Gen Z slang, TikTok vibes, trendy language"
        },
        "default_style": CONVERSATION_STYLE,
        "current_brand": BRAND_NAME
    }
# new addition for conversation style preview endpoint
@app.post("/conversation-styles/preview")
async def preview_conversation_style(
    style: str,
    sample_message: str = "Hello! I'd like to order something."
):
    """
    Preview how a conversation style affects responses.
    
    Args:
        style: The conversation style to preview ("default", "casual", "genz")
        sample_message: Sample message to generate response for
    
    Returns:
        dict: Sample response in the requested style
    """
    from fastapi import HTTPException
    
    # Validate style
    valid_styles = [s.value for s in ConversationStyle]
    if style not in valid_styles:
        raise HTTPException(status_code=400, detail=f"Invalid style. Choose from: {valid_styles}")
    
    # Create sample chat history
    chat_history = [
        Message(role="user", content=sample_message)
    ]
    
    # Generate preview response using assistant flow
    flow = get_conversation_flow(OrderAssistantFlowSK, style)
    
    return StreamingResponse(
        flow(
            chat_history=chat_history,
            current_order={},
            model_deployment=None,
        ),
        media_type="text/plain",
    )

@app.post("/clear-flow-cache")
async def clear_flow_cache():
    """Clear the conversation flow cache to force recreation with new styles."""
    global _flow_cache
    cache_size = len(_flow_cache)
    _flow_cache.clear()
    logging.info(f"Cleared {cache_size} cached flows")
    return {"status": "success", "cleared_flows": cache_size}
