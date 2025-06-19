from pathlib import Path
import json
import asyncio
from typing import Dict, List
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments

from streaming_ordering_chatbot.api.flows.classification_flow_SK import OrderIntentFlowSK
from streaming_ordering_chatbot.api.models import Message

class IntentEvaluationPlugin:
    """Semantic Kernel plugin for evaluating intent classification."""
    
    @kernel_function(name="evaluate_accuracy", description="Evaluates the accuracy of intent classification")
    def evaluate_accuracy(self, expected: str, predicted: str) -> float:
        return 1.0 if expected == predicted else 0.0
    
    @kernel_function(name="log_error", description="Logs classification errors with details")
    def log_error(self, message: str, expected: str, predicted: str, scenario: str) -> str:
        return f"Error in {scenario}: Expected {expected}, got {predicted} for message: '{message}'"

class SKIntentEvaluator:
    def __init__(self, endpoint: str, api_key: str, deployment_name: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
        
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
        self.classifier = OrderIntentFlowSK(endpoint, api_key, deployment_name)
    
    async def __call__(self, test_cases: List[Dict] = None) -> Dict:
        """
        Make the evaluator directly callable, similar to OrderIntentFlowSK.
        Args:
            test_cases: Optional list of test cases. If not provided, will load from default file.
                Each test case should be a dict with:
                - message: str
                - expected_intent: str
                - scenario: str
                - current_order: dict
        Returns:
            Dict containing:
            - metrics: Dict with evaluation metrics
                - accuracy: overall accuracy
                - f1_score: F1 score
                - precision: precision
                - recall: recall
                - per_intent: detailed metrics per intent
                - errors: list of misclassified cases
            - results_df: pandas DataFrame with detailed results per test case
        """
        if test_cases is None:
            # Load default test cases if none provided
            test_data_path = Path(__file__).parent / "data" / "intent_test_cases.json"
            with open(test_data_path, "r") as f:
                test_cases = json.load(f)["test_cases"]
        
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
                result["correct"] = False  # Make sure correct is False when prediction doesn't match
            else:
                error_logs[i] = None
                result["correct"] = True  # Make sure correct is True when prediction matches
                
            results.append(result)
        
        # Convert results to DataFrame
        df = pd.DataFrame(results)
        
        # Rename columns first to match desired format
        df = df.rename(columns={
            'expected': 'expected_intent',
            'predicted': 'predicted_intent'
        })
        
        # Add accuracy column (1.0 for correct predictions, 0.0 for incorrect)
        df['accuracy'] = (df['expected_intent'] == df['predicted_intent']).astype(float)
        
        # Add error_log column
        df['error_log'] = pd.Series(error_logs)
        
        # Select and order columns
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
        precisions, recalls, f1s, supports = precision_recall_fscore_support(
            y_true, y_pred, labels=unique_intents
        )
        
        # Create metrics output
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1_macro,
            'precision': precision_macro,
            'recall': recall_macro,
            'per_intent': {},
            'errors': []
        }
        
        # Add per-intent metrics
        for i, intent in enumerate(unique_intents):
            metrics['per_intent'][intent] = {
                'precision': precisions[i],
                'recall': recalls[i],
                'f1_score': f1s[i],
                'support': supports[i]
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
        '''
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
        
        # If prediction was wrong, log the error
        error_log = None
        if accuracy == 0:
            error_log = await self.kernel.invoke(
                plugin_name="evaluation",
                function_name="log_error",
                arguments=KernelArguments(
                    message=test_case["message"],
                    expected=test_case["expected_intent"],
                    predicted=predicted_intent,
                    scenario=test_case["scenario"]
                )
            )
        
        # Return complete result dictionary
        return {
            "message": test_case["message"],
            #"scenario": test_case["scenario"],
            "expected_intent": test_case["expected_intent"],
            "predicted_intent": predicted_intent,
            "accuracy": accuracy,
            "error_log": error_log
        }
        '''    

async def main():
    # Replace with your Azure OpenAI details
    ENDPOINT = ""
    API_KEY = ""
    DEPLOYMENT_NAME = "gpt-4o"
    
    # Create evaluator
    evaluator = SKIntentEvaluator(ENDPOINT, API_KEY, DEPLOYMENT_NAME)
    
    # Example 1: Evaluate using default test cases
    print("Running evaluation with default test cases...")
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
