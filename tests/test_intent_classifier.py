import json
from pathlib import Path
import asyncio
from typing import List, Dict
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from streaming_ordering_chatbot.api.flows.classification_flow_SK import OrderIntentFlowSK
from streaming_ordering_chatbot.api.models import Message

class IntentClassifierEvaluator:
    def __init__(self, endpoint: str, api_key: str, deployment_name: str):
        self.flow = OrderIntentFlowSK(endpoint, api_key, deployment_name)
        self.test_cases = self._load_test_cases()
        
    def _load_test_cases(self) -> List[Dict]:
        test_data_path = Path(__file__).parent / "data" / "intent_test_cases.json"
        with open(test_data_path, "r") as f:
            return json.load(f)["test_cases"]
    
    async def evaluate_single_case(self, test_case: Dict) -> Dict:
        """Evaluate a single test case and return the results."""
        chat_history = [Message(role="user", content=test_case["message"])]
        predicted_intent = await self.flow(chat_history, test_case["current_order"])
        
        return {
            "message": test_case["message"],
            "scenario": test_case["scenario"],
            "expected": test_case["expected_intent"],
            "predicted": predicted_intent,
            "correct": test_case["expected_intent"] == predicted_intent
        }
    
    async def evaluate_all(self) -> Dict:
        """Run evaluation on all test cases and compute metrics."""
        results = []
        for case in self.test_cases:
            result = await self.evaluate_single_case(case)
            results.append(result)
        
        df = pd.DataFrame(results)
        
        y_true = df["expected"].tolist()
        y_pred = df["predicted"].tolist()
        
        report = classification_report(y_true, y_pred, output_dict=True)
        
        cm = confusion_matrix(y_true, y_pred)
        
        scenario_accuracy = df.groupby("scenario")["correct"].mean()
        
        errors = df[~df["correct"]]
        
        return {
            "detailed_results": df.to_dict(orient="records"),
            "classification_report": report,
            "confusion_matrix": cm,
            "scenario_accuracy": scenario_accuracy.to_dict(),
            "error_cases": errors.to_dict(orient="records"),
            "overall_accuracy": df["correct"].mean()
        }
    
    def plot_confusion_matrix(self, cm):
        """Plot and save confusion matrix."""
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=["conversation", "order"],
                   yticklabels=["conversation", "order"])
        plt.title("Confusion Matrix")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.savefig("confusion_matrix.png")
        plt.close()

async def main():
    ENDPOINT = ""
    API_KEY = ""
    DEPLOYMENT_NAME = "gpt-4o"
    
    evaluator = IntentClassifierEvaluator(ENDPOINT, API_KEY, DEPLOYMENT_NAME)
    results = await evaluator.evaluate_all()
    
    print("\n=== Overall Results ===")
    print(f"Accuracy: {results['overall_accuracy']:.2%}")
    
    print("\n=== Classification Report ===")
    report = pd.DataFrame(results['classification_report']).drop('support', axis=0)
    print(report)
    
    print("\n=== Accuracy by Scenario ===")
    for scenario, acc in results['scenario_accuracy'].items():
        print(f"{scenario}: {acc:.2%}")
    
    print("\n=== Error Cases ===")
    for error in results['error_cases']:
        print(f"Message: {error['message']}")
        print(f"Expected: {error['expected']}, Got: {error['predicted']}")
        print(f"Scenario: {error['scenario']}\n")
    
    # Plot confusion matrix
    evaluator.plot_confusion_matrix(results['confusion_matrix'])

if __name__ == "__main__":
    asyncio.run(main())
