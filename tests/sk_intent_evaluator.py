from pathlib import Path
import json
import asyncio
import os
from typing import Dict, List, Optional
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from dotenv import load_dotenv
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments

from streaming_ordering_chatbot.api.flows.classification_flow_SK import OrderIntentFlowSK
from streaming_ordering_chatbot.api.models import Message

load_dotenv()

class IntentEvaluationPlugin:
    """Semantic Kernel plugin for evaluating intent classification."""
    
    @kernel_function(name="evaluate_accuracy", description="Evaluates the accuracy of intent classification")
    def evaluate_accuracy(self, expected: str, predicted: str) -> float:
        return 1.0 if expected == predicted else 0.0
    
    @kernel_function(name="log_error", description="Logs classification errors with details")
    def log_error(self, message: str, expected: str, predicted: str, scenario: str) -> str:
        return f"Error in {scenario}: Expected {expected}, got {predicted} for message: '{message}'"

class SKIntentEvaluator:
    def __init__(self, endpoint: str, api_key: str, deployment_name: str, brand_name: str = "default"):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.brand_name = brand_name

        # Initialize Semantic Kernel
        self.kernel = Kernel()
        chat_service = AzureChatCompletion(
            deployment_name=self.deployment_name,
            endpoint=self.endpoint,
            api_key=self.api_key
        )
        self.kernel.add_service(chat_service)
        
        # Add evaluation plugin
        self.kernel.add_plugin(IntentEvaluationPlugin(), "evaluation")
        
        # Initialize classifier
        self.classifier = OrderIntentFlowSK(endpoint, api_key, deployment_name, self.brand_name)
    
    async def __call__(self, test_cases: Optional[List[Dict]] = None) -> Dict:
        """
        Evaluate intent classification using provided test cases or default cases.
        """
        if test_cases is None:
            # Load default test cases if none provided
            test_data_path = Path(__file__).parent / "data" / "intent_test_cases.json"
            with open(test_data_path, "r") as f:
                test_cases = json.load(f)["test_cases"]
        
        # Ensure test_cases is not None
        test_cases = test_cases or []
        
        results = []
        error_logs = {}  # Store error logs by index
        
        for i, case in enumerate(test_cases):
            result = await self.evaluate_single_case(case)
            
            # Generate error log if prediction was wrong
            if result["predicted"] != result["expected"]:
                error_log = await self.kernel.invoke(
                    plugin_name="evaluation",
                    function_name="log_error",
                    arguments=KernelArguments(
                        message=result["message"],
                        expected=result["expected"],
                        predicted=result["predicted"],
                        scenario=result["scenario"]
                    )
                )
                error_logs[i] = error_log
                result["correct"] = False  
            else:
                error_logs[i] = None
                result["correct"] = True  
                
            results.append(result)
        
        df = pd.DataFrame(results)
        
        df = df.rename(columns={
            'expected': 'expected_intent',
            'predicted': 'predicted_intent'
        })
        
        df['accuracy'] = (df['expected_intent'] == df['predicted_intent']).astype(float)
        
        df['error_log'] = pd.Series(error_logs)
        
        df = df[['message', 'scenario', 'expected_intent', 'predicted_intent', 'accuracy', 'error_log']]
        
        # Calculate metrics using sklearn
        y_true = df['expected_intent'].tolist()
        y_pred = df['predicted_intent'].tolist()
        
        # Calculate overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            y_true, y_pred, average='macro'
        )
        
        # Calculate per-intent metrics
        unique_intents = sorted(set(df['expected_intent']))
        precision_dict = {}
        recall_dict = {}
        f1_dict = {}
        support_dict = {}
        
        for intent in unique_intents:
            # Calculate metrics for each intent separately
            y_true_intent = [1 if y == intent else 0 for y in y_true]
            y_pred_intent = [1 if y == intent else 0 for y in y_pred]
            precision, recall, f1, support = precision_recall_fscore_support(
                y_true_intent, y_pred_intent, average='binary'
            )
            precision_dict[intent] = precision
            recall_dict[intent] = recall
            f1_dict[intent] = f1
            support_dict[intent] = sum(y_true_intent)
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1_macro,
            'precision': precision_macro,
            'recall': recall_macro,
            'per_intent': {},
            'errors': []
        }
        
        for intent in unique_intents:
            metrics['per_intent'][intent] = {
                'precision': precision_dict[intent],
                'recall': recall_dict[intent],
                'f1_score': f1_dict[intent],
                'support': support_dict[intent]
            }
        
        # Add error cases
        errors = df[df['accuracy'] == 0].to_dict('records')
        metrics['errors'] = [{
            'message': e['message'],
            'expected': e['expected_intent'],
            'predicted': e['predicted_intent'],
            'scenario': e['scenario']
        } for e in errors]
        
        return {
            'metrics': metrics,
            'results_df': df
        }
    
    async def evaluate_single_case(self, test_case: Dict) -> Dict:
        """Evaluate a single test case using Semantic Kernel functions."""
        chat_history = [Message(role="user", content=test_case["message"])]
        predicted_intent = await self.classifier(chat_history, test_case["current_order"])
        
        # Use SK function to evaluate accuracy
        accuracy = await self.kernel.invoke(
            plugin_name="evaluation",
            function_name="evaluate_accuracy",
            arguments=KernelArguments(
                expected=test_case["expected_intent"],
                predicted=predicted_intent
            )
        )
        
        return {
            "message": test_case["message"],
            "scenario": test_case["scenario"],
            "expected": test_case["expected_intent"],
            "predicted": predicted_intent,
            "correct": bool(accuracy)
        }
    
async def main():
    ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    BRAND_NAME = os.getenv("RESTAURANT_BRAND")
    
    if not all([ENDPOINT, API_KEY, DEPLOYMENT_NAME]):
        raise ValueError("Missing required environment variables")
    
    # Create evaluator
    evaluator = SKIntentEvaluator(str(ENDPOINT), str(API_KEY), str(DEPLOYMENT_NAME), str(BRAND_NAME))
    
    print("Running evaluation with default test cases for brand: {BRAND_NAME}")
    results = await evaluator()
    
    # Print metrics
    print("\nOverall Metrics:")
    print(f"Accuracy: {results['metrics']['accuracy']:.2%}")
    print(f"F1 Score: {results['metrics']['f1_score']:.2%}")
    print(f"Precision: {results['metrics']['precision']:.2%}")
    print(f"Recall: {results['metrics']['recall']:.2%}")
    
    print("\nPer-Intent Metrics:")
    for intent, metrics in results['metrics']['per_intent'].items():
        print(f"\n{intent}:")
        print(f"  Precision: {metrics['precision']:.2%}")
        print(f"  Recall: {metrics['recall']:.2%}")
        print(f"  F1 Score: {metrics['f1_score']:.2%}")
        print(f"  Support: {metrics['support']} samples")
    
    if results['metrics']['errors']:
        print("\nMisclassified Cases:")
        for error in results['metrics']['errors']:
            print(f"\nMessage: {error['message']}")
            print(f"Expected: {error['expected']}, Got: {error['predicted']}")
            print(f"Scenario: {error['scenario']}")

if __name__ == "__main__":
    asyncio.run(main())
