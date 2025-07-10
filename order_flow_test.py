import asyncio
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import os

from streaming_ordering_chatbot.api.flows.order_flow_SK import OrderFlowSK
from streaming_ordering_chatbot.api.models import Message

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv()

def get_required_env_var(name: str) -> str:
    """Get a required environment variable or raise an informative error."""
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

BRAND_NAME = get_required_env_var("RESTAURANT_BRAND")


class OrderFlowTester:
    """Test class for OrderFlowSK functionality."""
    
    def __init__(self, endpoint: str, api_key: str, deployment_name: str, brand_name: str):
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.brand_name = brand_name
        self.order_flow = OrderFlowSK(endpoint, api_key, deployment_name, brand_name)
        self.test_results = []
    
    def get_test_cases(self) -> List[Dict[str, Any]]:
        """Load test cases from JSON file.""" 
        test_data_path = Path(__file__).parent.parent / "Ordering_ChatBot" / "tests" / "data" / "order_test_cases.json"
        
        try:
            with open(test_data_path, "r") as f:
                test_data = json.load(f)
            
            # Convert test cases to expected format
            test_cases = []
            for case in test_data["test_cases"]:
                test_case = {
                    "name": case["name"],
                    "description": case["description"],
                    "chat_history": [Message(role="user", content=case["message"])],
                    "current_order": case["current_order"],
                    "expected_valid": case["expected_valid"],
                    "expected_items": case["expected_items"],
                    "category": case.get("category", "unknown")
                }
                test_cases.append(test_case)
            
            return test_cases
            
        except FileNotFoundError:
            logger.warning(f"Test data file not found: {test_data_path}")
            # Return a minimal set of test cases as fallback
            return []
    
    async def run_single_test(self, test_case: Dict[str, Any], use_streaming: bool = False) -> Dict[str, Any]:
        """Run a single test case."""
        logger.info(f"Running test: {test_case['name']} (streaming: {use_streaming})")
        
        try:
            # Process the order
            if use_streaming:
                result_chunks = []
                async for chunk in self.order_flow(
                    test_case["chat_history"], 
                    test_case["current_order"], 
                    use_streaming=True
                ):
                    result_chunks.append(chunk)
                
                # Combine streaming chunks
                raw_response = "".join(result_chunks)
            else:
                # Non-streaming mode
                result_generator = self.order_flow(
                    test_case["chat_history"], 
                    test_case["current_order"], 
                    use_streaming=False
                )
                raw_response = ""
                async for chunk in result_generator:
                    raw_response += chunk
            
            # Parse the response
            try:
                # Try to parse as JSON first
                if raw_response.strip().startswith('{'):
                    parsed_response = json.loads(raw_response.strip())
                else:
                    # If not JSON, treat as text response
                    parsed_response = {"text_response": raw_response.strip()}
            except json.JSONDecodeError:
                parsed_response = {"text_response": raw_response.strip()}
            
            # Analyze the result
            is_valid_order = self._analyze_order_validity(parsed_response)
            extracted_items = self._extract_items_from_response(parsed_response)
            
            # Create result record
            result = {
                "input_message": test_case["chat_history"][-1].content,
                "expected_items": test_case["expected_items"],
                "response": parsed_response,
                "positive": (extracted_items == test_case["expected_items"]),
                "timestamp": datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error in test {test_case['chat_history'][-1].content}: {e}")
            return {
                "input_message": test_case["chat_history"][-1].content,
                "expected_items": test_case["expected_items"],
                "extracted_items": extracted_items,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _analyze_order_validity(self, parsed_response: Dict[str, Any]) -> bool:
        """Analyze if the response represents a valid order."""
        if "error" in parsed_response:
            return False
        
        # Check if response contains order items
        if "items" in parsed_response and isinstance(parsed_response["items"], list):
            return len(parsed_response["items"]) > 0
        
        # Check if response contains structured order data
        if any(key in parsed_response for key in ["burgers", "drinks", "fries", "sides"]):
            return True
        
        # Check text response for order indicators
        if "text_response" in parsed_response:
            text = parsed_response["text_response"].lower()
            order_indicators = ["order", "burger", "fries", "drink", "item", "add", "remove", "modify"]
            return any(indicator in text for indicator in order_indicators)
        
        return False
    
    def _extract_items_from_response(self, parsed_response: Dict[str, Any]) -> List[str]:
        """Extract item names from the response."""
        items = []
        
        # Extract from structured items list
        if "items" in parsed_response and isinstance(parsed_response["items"], list):
            for item in parsed_response["items"]:
                if isinstance(item, dict) and "name" in item:
                    items.append(item["name"].lower())
        
        # Extract from categorized items
        for category in ["burgers", "drinks", "fries", "sides"]:
            if category in parsed_response and isinstance(parsed_response[category], list):
                for item in parsed_response[category]:
                    if isinstance(item, dict) and "name" in item:
                        items.append(item["name"].lower())
        
        return items
    
    async def run_all_tests(self, test_streaming: bool = True) -> Dict[str, Any]:
        """Run all test cases."""
        logger.info("Starting comprehensive order flow tests...")
        
        test_cases = self.get_test_cases()
        all_results = []
        
        for test_case in test_cases:
            if test_streaming:
                result = await self.run_single_test(test_case, use_streaming=True)
            #result = await self.run_single_test(test_case, use_streaming=False)
                all_results.append(result)

        results = {
            "results": all_results
        }
        return results
    
    def save_results(self, results: Dict[str, Any], output_dir: str = "evaluation_results/order_evaluation") -> str:
        """Save test results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Save detailed results as JSON
        json_file = output_path / f"order_flow_test_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary as CSV
        csv_file = output_path / f"order_flow_test_summary_{timestamp}.csv"
        results_df = pd.DataFrame(results["results"])
        results_df.to_csv(csv_file, index=False)
        
        logger.info(f"Results saved to {json_file} and {csv_file}")
        return str(json_file)
    
async def run_interactive_test():
    """Run an interactive test with user input."""
    print("\n" + "="*60)
    print("INTERACTIVE ORDER FLOW TEST")
    print("="*60)
    print("Enter an order to test (or 'quit' to exit):")
    
    tester = OrderFlowTester(ENDPOINT, API_KEY, DEPLOYMENT_NAME, BRAND_NAME)
    
    while True:
        user_input = input("\nYour order: ").strip()
        if user_input.lower() in ['quit', 'exit', 'q']:
            break
        
        if not user_input:
            continue
        
        # Create test case from user input
        chat_history = [Message(role="user", content=user_input)]
        current_order = {"items": []}
        
        print(f"\nProcessing order: '{user_input}'")
        print("-" * 40)
        
        # Test both modes
        for streaming in [False, True]:
            mode = "Streaming" if streaming else "Non-Streaming"
            print(f"\n{mode} Mode:")
            
            try:
                result_chunks = []
                async for chunk in tester.order_flow(chat_history, current_order, use_streaming=streaming):
                    result_chunks.append(chunk)
                    if streaming:
                        print(chunk, end='', flush=True)
                
                if not streaming:
                    print("".join(result_chunks))
                
            except Exception as e:
                print(f"Error: {e}")
        
        print("\n" + "-" * 40)


async def main():
    """Main test function."""
    if not ENDPOINT or not API_KEY or not DEPLOYMENT_NAME:
        print(" Missing required environment variables:")
        print("   AZURE_OPENAI_ENDPOINT")
        print("   AZURE_OPENAI_API_KEY") 
        print("   AZURE_OPENAI_DEPLOYMENT_NAME")
        print("Please set these variables in your .env file.")
        return
    
    print("Order Flow Test")
    print("=" * 50)
    print(f"Testing brand: {BRAND_NAME}")
    
    try:
        # Initialize tester
        tester = OrderFlowTester(ENDPOINT, API_KEY, DEPLOYMENT_NAME, BRAND_NAME)
        
        # Run comprehensive tests
        print("\nRunning comprehensive tests...")
        results = await tester.run_all_tests(test_streaming=True)
    
        # Save results
        output_file = tester.save_results(results)
        print(f"\nDetailed results saved to: {output_file}")
        
        # Run interactive test
        await run_interactive_test()
        
    except Exception as e:
        print(f"Error initializing test case: {e}")
        print("Make sure:")
        print("1. Environment variables are set correctly")
        print("2. The specified brand is configured with a menu file")
        print("3. Azure OpenAI credentials are valid")


if __name__ == "__main__":
    asyncio.run(main())
