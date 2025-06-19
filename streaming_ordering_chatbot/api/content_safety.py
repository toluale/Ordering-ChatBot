import asyncio
import os
from typing import AsyncGenerator, Optional, Any, List, cast

import httpx
from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions
from azure.core.credentials import AzureKeyCredential
from presidio_analyzer import AnalyzerEngine
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig, RecognizerResult

def get_required_env_var(name: str) -> str:
    """Get a required environment variable or raise an informative error."""
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"{name} environment variable is not set. "
            "Please set it to the appropriate value."
        )
    return value

# Content safety configuration
AZURE_CONTENT_SAFETY_KEY ="8cfBQF1HE4qzxIn5VapNbWeqhqpYIR6OnHq0zXvxp3gVOz3YC2uOJQQJ99BFACHYHv6XJ3w3AAAAACOGGDMG" #get_required_env_var("AZURE_CONTENT_SAFETY_KEY")
AZURE_CONTENT_SAFETY_ENDPOINT ="https://t-toluale-1040-resource.openai.azure.com/" #get_required_env_var("AZURE_CONTENT_SAFETY_ENDPOINT")
BLOCKLIST_NAME = "CustomBlocklist296" #get_required_env_var("BLOCKLIST_NAME")

# Content safety severity limits
CATEGORY_SEVERITY_LIMITS = {
    "Hate": int(os.getenv("HATE_SEVERITY_LIMIT", "1")),
    "SelfHarm": int(os.getenv("SELF_HARM_SEVERITY_LIMIT", "1")),
    "Sexual": int(os.getenv("SEXUAL_SEVERITY_LIMIT", "1")),
    "Violence": int(os.getenv("VIOLENCE_SEVERITY_LIMIT", "1")),
}

# Initialize clients
analyzer = AnalyzerEngine()
anonymizer = AnonymizerEngine()
moderation_client = ContentSafetyClient(
    endpoint=AZURE_CONTENT_SAFETY_ENDPOINT,
    credential=AzureKeyCredential(AZURE_CONTENT_SAFETY_KEY),
)

async def filter_text(chunk_text: str) -> Optional[str]:
    """Filter sensitive PII from text.

    Args:
        chunk_text (str): Text to filter.

    Returns:
        Optional[str]: Filtered text if PII was found, None otherwise.
    """
    analyzer_results = analyzer.analyze(text=chunk_text, language="en")
    if not analyzer_results:
        return None

    # Convert analyzer results to anonymizer format
    anonymizer_results = []
    for result in analyzer_results:
        result_dict = result.to_dict()
        # Keep only essential fields
        supported_fields = {
            'entity_type': result_dict['entity_type'],
            'start': result_dict['start'],
            'end': result_dict['end'],
            'score': result_dict['score']
        }
        anonymizer_results.append(RecognizerResult(**supported_fields))

    if not anonymizer_results:
        return None

    # Apply anonymization
    anonymized = anonymizer.anonymize(
        text=chunk_text,
        analyzer_results=anonymizer_results,
        operators={
            "DEFAULT": OperatorConfig("keep"),
            "PHONE_NUMBER": OperatorConfig(
                "replace", {"new_value": "<REDACTED PHONE NUMBER>"}
            ),
            "CREDIT_CARD": OperatorConfig(
                "replace", {"new_value": "<REDACTED CREDIT CARD>"}
            ),
        },
    )
    return anonymized.text if len(anonymized.items) > 0 else None

def moderate_text(text: str) -> List[str]:
    """Check text against content safety rules.

    Args:
        text (str): Text to moderate.

    Returns:
        List[str]: List of failed category names.
    """
    request = AnalyzeTextOptions(text=text, blocklist_names=[BLOCKLIST_NAME])
    response = moderation_client.analyze_text(request)

    failed_categories = []
    for category in response.categories_analysis:
        severity = category.severity or 0
        if severity > CATEGORY_SEVERITY_LIMITS[category.category]:
            failed_categories.append(category.category)
    return failed_categories

async def pre_process_check(user_message: str):
    """Pre-process user message for content safety.

    Args:
        user_message (str): Raw user message.

    Returns:
        Tuple[str, List[str]]: (Processed message, List of failed categories)
    """
    # First check for PII and filter if needed
    if filtered_message := await filter_text(user_message):
        user_message = filtered_message

    # Then check content safety
    failed_categories = moderate_text(user_message)
    return user_message, failed_categories

def wrap_content_safety(generator_func, validation_interval: int = 5):
    """Wrap an async generator with content safety checks.

    Args:
        generator_func: Async generator function to wrap.
        validation_interval (int): How often to check content (in chunks).

    Returns:
        AsyncGenerator[str, None]: Safe content generator.
    """
    async def get_text_content(token: Any) -> Optional[str]:
        """Extract text content from a token."""
        if hasattr(token, 'items') and token.items:
            for item in token.items:
                if hasattr(item, 'text') and item.text is not None:
                    return item.text.strip()
        elif isinstance(token, str):
            return token.strip()
        return None

    async def safety_redaction(*args: Any, **kwargs: Any) -> AsyncGenerator[str, None]:
        n = -1
        try:
            async for token in generator_func(*args, **kwargs):
                n += 1
                text_content = await get_text_content(token)
                
                if text_content:
                    # Check content safety periodically
                    if n % validation_interval == 0:
                        try:
                            failed_categories = moderate_text(text_content)
                            if failed_categories:
                                continue  # Skip unsafe content
                        except Exception as e:
                            print(f"Content safety error: {str(e)}", flush=True)
                            continue
                    yield text_content

        except Exception as e:
            print(f"Streaming error: {str(e)}", flush=True)
            raise

    return safety_redaction
