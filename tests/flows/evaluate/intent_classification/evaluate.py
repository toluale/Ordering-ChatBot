from typing import Dict
import json
from pathlib import Path

from promptflow import tool
from promptflow.contracts.types import ToolType
from promptflow.contracts.flow import FlowContext
from promptflow.core import log_metric

from streaming_ordering_chatbot.api.flows.classification_flow_SK import OrderIntentFlowSK
from streaming_ordering_chatbot.api.models import Message

class IntentClassifierEvaluator:
    def __init__(self, flow_context: FlowContext):
        self.context = flow_context
        # Get credentials from flow context
        self.endpoint = flow_context.get_secret("azure_openai_endpoint")
        self.api_key = flow_context.get_secret("azure_openai_api_key")
        self.deployment_name = flow_context.get_secret("deployment_name", "gpt-4o")
        self.flow = OrderIntentFlowSK(self.endpoint, self.api_key, self.deployment_name)

    async def evaluate_single(self, test_case: Dict) -> Dict:
        chat_history = [Message(role="user", content=test_case["message"])]
        predicted_intent = await self.flow(chat_history, test_case["current_order"])
        
        is_correct = test_case["expected_intent"] == predicted_intent
        
        return {
            "message": test_case["message"],
            "scenario": test_case["scenario"],
            "expected_intent": test_case["expected_intent"],
            "predicted_intent": predicted_intent,
            "is_correct": is_correct
        }

@tool(type=ToolType.PYTHON, name="evaluate_intent_classification")
async def evaluate_intent(context: FlowContext, test_case: Dict) -> Dict:
    """
    Evaluate a single test case for intent classification.
    """
    evaluator = IntentClassifierEvaluator(context)
    result = await evaluator.evaluate_single(test_case)
    
    # Log metrics for this test case
    log_metric(
        f"accuracy_{result['scenario']}", 
        1.0 if result['is_correct'] else 0.0
    )
    
    # Log detailed results
    log_metric("test_case_details", {
        "message": result["message"],
        "scenario": result["scenario"],
        "expected": result["expected_intent"],
        "predicted": result["predicted_intent"],
        "correct": result["is_correct"]
    })
    
    return result
