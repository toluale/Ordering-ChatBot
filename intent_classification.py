import asyncio
import json
from pathlib import Path
from streaming_ordering_chatbot.api.flows.classification_flow_SK import OrderIntentFlowSK
from streaming_ordering_chatbot.api.models import Message
import pandas as pd
from tests.sk_intent_evaluator import SKIntentEvaluator

# Azure OpenAI configuration
ENDPOINT = "https://t-toluale-1040-resource.openai.azure.com/"
API_KEY = "8cfBQF1HE4qzxIn5VapNbWeqhqpYIR6OnHq0zXvxp3gVOz3YC2uOJQQJ99BFACHYHv6XJ3w3AAAAACOGGDMG"
DEPLOYMENT_NAME = "gpt-4o"

async def test_order_intent():
    """Test individual intent classification"""
    chat_history = [Message(role="user", content="What toppings do you have?.")]
    current_order = {"items": [{"name": "fries", "quantity": 1}]}
    
    flow = OrderIntentFlowSK(ENDPOINT, API_KEY, DEPLOYMENT_NAME)
    result = await flow(chat_history, current_order)
    print("Intent classification result:", result)

async def evaluate_intent_classifier():
    """Run full evaluation of the intent classifier"""
    evaluator = SKIntentEvaluator(ENDPOINT, API_KEY, DEPLOYMENT_NAME)
    
    # Load test cases to show alongside results
    test_data_path = Path(__file__).parent.parent / "streaming-ordering-chatbot" / "tests" / "data" / "intent_test_cases.json"
    with open(test_data_path, "r") as f:
        test_cases = json.load(f)["test_cases"]
    #C:\Users\t-toluale\.vscode\streaming-ordering-chatbot\tests\data\intent_test_cases.json
    # Run evaluation with default test cases
    metrics = await evaluator()
    
    print("\nEvaluation Metrics:")
    print(f"Accuracy: {metrics['metrics']['accuracy']:.2f}")
    print(f"F1 Score: {metrics['metrics']['f1_score']:.2f}")
    print(f"Precision: {metrics['metrics']['precision']:.2f}")
    print(f"Recall: {metrics['metrics']['recall']:.2f}")
    
    print("\nPer-Intent Metrics:")
    for intent, scores in metrics['metrics']['per_intent'].items():
        print(f"\n{intent}:")
        for metric, value in scores.items():
            print(f"  {metric}: {value:.2f}")
    
    print("\nMisclassified Cases:")
    if 'errors' in metrics['metrics']:
        for error in metrics['metrics']['errors']:
            print(f"\nError: {error}")
    else:
        print("No misclassified cases found.")
    
    # Log file location
    print("\nDetailed logs can be found in: streaming_ordering_chatbot.conversation_flow_sk.log")
    
    return metrics

async def main():
    # Test individual intent classification
    print("Testing individual intent classification:")
    await test_order_intent()
    
    # Run full evaluation
    print("\nRunning full evaluation:")
    await evaluate_intent_classifier()

if __name__ == "__main__":
    asyncio.run(main())