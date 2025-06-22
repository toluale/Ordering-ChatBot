import os
import asyncio
import json
from pathlib import Path
from streaming_ordering_chatbot.api.flows.classification_flow_SK import OrderIntentFlowSK
from streaming_ordering_chatbot.api.models import Message
import pandas as pd
from dotenv import load_dotenv
from tests.sk_intent_evaluator import SKIntentEvaluator

# Load environment variables from .env file
load_dotenv()

def get_required_env_var(name: str) -> str:
    """Get a required environment variable or raise an informative error.
    
    Args:
        name (str): Name of the environment variable
    
    Returns:
        str: Value of the environment variable
        
    Raises:
        ValueError: If the environment variable is not set or is empty
    """
    value = os.getenv(name)
    if not value:
        raise ValueError(
            f"{name} environment variable is not set. "
            "Please check your .env file."
        )
    return str(value)

# Azure OpenAI configuration
ENDPOINT = get_required_env_var("AZURE_OPENAI_ENDPOINT")
API_KEY = get_required_env_var("AZURE_OPENAI_API_KEY")
DEPLOYMENT_NAME = get_required_env_var("AZURE_OPENAI_DEPLOYMENT_NAME")

async def test_order_intent() -> str:
    """Test individual intent classification with a sample message.
    
    Returns:
        str: The classified intent
    """
    chat_history = [Message(role="user", content="What toppings do you have?")]
    current_order = {"items": [{"name": "fries", "quantity": 1}]}
    
    try:
        flow = OrderIntentFlowSK(ENDPOINT, API_KEY, DEPLOYMENT_NAME)
        result = await flow(chat_history, current_order)
        print("Intent classification result:", result)
        return result
    except Exception as e:
        print(f"Error during intent classification: {e}")
        raise

async def evaluate_intent_classifier() -> dict:
    """Run full evaluation of the intent classifier using test cases.
    
    Returns:
        dict: Evaluation metrics including accuracy, F1 score, precision, and recall
    """
    try:
        evaluator = SKIntentEvaluator(ENDPOINT, API_KEY, DEPLOYMENT_NAME)
        
        # Load test cases
        test_data_path = Path(__file__).parent / "tests" / "data" / "intent_test_cases.json"
        if not test_data_path.exists():
            raise FileNotFoundError(f"Test cases file not found at {test_data_path}")
            
        with open(test_data_path, "r") as f:
            test_cases = json.load(f)["test_cases"]
        
        # Run evaluation with test cases
        metrics = await evaluator()
        
        # Print overall metrics
        print("\nEvaluation Metrics:")
        print(f"Accuracy: {metrics['metrics']['accuracy']:.2%}")
        print(f"F1 Score: {metrics['metrics']['f1_score']:.2%}")
        print(f"Precision: {metrics['metrics']['precision']:.2%}")
        print(f"Recall: {metrics['metrics']['recall']:.2%}")
        
        # Print per-intent metrics
        print("\nPer-Intent Metrics:")
        for intent, scores in metrics['metrics']['per_intent'].items():
            print(f"\n{intent}:")
            for metric, value in scores.items():
                if isinstance(value, float):
                    print(f"  {metric}: {value:.2%}")
                else:
                    print(f"  {metric}: {value}")
        
        # Print misclassified cases
        print("\nMisclassified Cases:")
        if metrics['metrics']['errors']:
            for error in metrics['metrics']['errors']:
                print(f"\nMessage: {error['message']}")
                print(f"Expected: {error['expected']}, Got: {error['predicted']}")
        else:
            print("No misclassified cases found.")
        
        print("\nDetailed logs can be found in: streaming_ordering_chatbot.conversation_flow_sk.log")
        
        return metrics
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

async def main():
    # Test individual intent classification
    print("Testing individual intent classification:")
    await test_order_intent()
    
    # Run full evaluation
    print("\nRunning full evaluation:")
    await evaluate_intent_classifier()

if __name__ == "__main__":
    asyncio.run(main())
