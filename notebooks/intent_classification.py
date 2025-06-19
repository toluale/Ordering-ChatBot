import asyncio
import json
from pathlib import Path
from streaming_ordering_chatbot.api.flows.classification_flow_SK import OrderIntentFlowSK
from streaming_ordering_chatbot.api.models import Message
import pandas as pd
from tests.sk_intent_evaluator import SKIntentEvaluator

# Replace with your actual Azure OpenAI details
ENDPOINT = "https://t-toluale-1040-resource.openai.azure.com/"
API_KEY = "8cfBQF1HE4qzxIn5VapNbWeqhqpYIR6OnHq0zXvxp3gVOz3YC2uOJQQJ99BFACHYHv6XJ3w3AAAAACOGGDMG"
DEPLOYMENT_NAME = "gpt-4o"

# Sample chat history and order
chat_history = [Message(role="user", content="What toppings do you have?.")]
current_order = {"items": [{"name": "fries", "quantity": 1}]}

async def test_order_intent():
    flow = OrderIntentFlowSK(ENDPOINT, API_KEY, DEPLOYMENT_NAME)
    result = await flow(chat_history, current_order)
    print("Intent classification result:", result)

async def evaluate_intent_classifier():
    evaluator = SKIntentEvaluator(ENDPOINT, API_KEY, DEPLOYMENT_NAME)
    
    # Load test cases to show alongside results
    test_data_path = Path(__file__).parent.parent / "tests" / "data" / "intent_test_cases.json"
    with open(test_data_path, "r") as f:
        test_cases = json.load(f)["test_cases"]
    
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
    
    # Save detailed results to a CSV file
    if 'results' in metrics:
        results_df = pd.DataFrame(metrics['results'])
        output_path = Path(__file__).parent / "intent_classification_results.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nDetailed results saved to: {output_path}")
    
    print("\nMisclassified Cases:")
    for result in metrics['results']:
        if result['expected'] != result['predicted']:
            print(f"\nScenario: {result['scenario']}")
            print(f"Message: {result['message']}")
            print(f"Expected: {result['expected']}")
            print(f"Predicted: {result['predicted']}")
    
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