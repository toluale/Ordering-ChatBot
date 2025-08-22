from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict
from openai import AzureOpenAI


DEFAULT_API_VERSION = "2024-12-01-preview"


def create_azure_openai_client(api_key: str, endpoint: str, api_version: Optional[str] = None) -> AzureOpenAI:
    """Create and return an AzureOpenAI client with a consistent API version.

    Args:
        api_key: Azure OpenAI API key
        endpoint: Azure OpenAI endpoint URL
        api_version: Optional api-version override (defaults to DEFAULT_API_VERSION)
    """
    return AzureOpenAI(
        api_key=api_key,
        api_version=api_version or DEFAULT_API_VERSION,
        azure_endpoint=endpoint,
    )


# Centralized defaults for chat completions
@dataclass(frozen=True)
class ChatCompletionDefaults:
    temperature: float = 0.75
    top_p: float = 0.95
    presence_penalty: float = 0.6
    frequency_penalty: float = 0.3
    stream: bool = True


DEFAULT_CHAT_COMPLETION_DEFAULTS = ChatCompletionDefaults()


def build_chat_params(
    overrides: Optional[Dict[str, Any]] = None,
    defaults: ChatCompletionDefaults = DEFAULT_CHAT_COMPLETION_DEFAULTS,
) -> Dict[str, Any]:
    """Build a dict of chat.completions.create parameters from centralized defaults with optional overrides.

    Note:
      - max_tokens is intentionally not set by default; pass it in overrides when needed.
      - stream defaults to True; override with {"stream": False} if a non-streaming call is desired.
    """
    params: Dict[str, Any] = asdict(defaults)
    if overrides:
        params.update({k: v for k, v in overrides.items() if v is not None})
    return params
