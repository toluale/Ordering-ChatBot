# conversation_brand_evaluator_v2.py

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, AsyncGenerator, Type, cast
import json
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from streaming_ordering_chatbot.api.utils.azure_client import create_azure_openai_client

from semantic_kernel.functions.kernel_arguments import KernelArguments

from streaming_ordering_chatbot.api.flows import (PreambleFlowSK, OrderAssistantFlowSK, SummaryFlowSK)
from streaming_ordering_chatbot.api.models import Message
from streaming_ordering_chatbot.evaluation.metrics import (EvaluationMetric, BrandVoiceMetric, RelevanceMetric, TaskCompletionMetric)
from streaming_ordering_chatbot.evaluation.scenarios import (ScenarioLoader, ScenarioConfig)


class BrandConfigValidationError(Exception):
    """Raised when brand configuration is invalid."""
    pass


class EvaluationStatus(Enum):
    """Status of an evaluation run."""
    SUCCESS = "success"
    PARTIAL_SUCCESS = "partial_success"
    FAILURE = "failure"


@dataclass
class EvaluationResult:
    """Results of a conversation evaluation."""
    scenario_id: str
    brand_name: str
    scores: Dict[str, float]
    conversation: List[Dict[str, str]]
    timestamp: datetime
    status: EvaluationStatus
    errors: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert result to a dictionary for serialization."""
        return {
            "scenario_id": self.scenario_id,
            "brand_name": self.brand_name,
            "scores": self.scores,
            "conversation": self.conversation,
            "timestamp": self.timestamp.isoformat(),
            "status": self.status.value,
            "errors": self.errors or []
        }


class BrandedChatbotEvaluator:
    """Evaluates branded chatbot conversations."""
    
    REQUIRED_BRAND_FIELDS = ["name", "tone", "style", "values"]
    DEFAULT_API_VERSION = "2024-12-01-preview"
    DEFAULT_RESULTS_DIR = "evaluation_results"
    
    async def _handle_conversation_turn(
        self,
        user_message: str,
        chat_history: List[Message],
        current_order: Dict[str, Any],
        flow_type: str = "order"
    ) -> str:
        """Handle a single turn in the conversation using the specified flow."""
        # Add user message to chat history
        user_msg = Message(content=user_message, role="user")
        chat_history.append(user_msg)
        
        # Get assistant response using direct call to appropriate SK flow
        response_generator = self.flows[flow_type](chat_history=chat_history, current_order=current_order)
        
        # Collect response chunks from the generator
        response_chunks = []
        async for chunk in response_generator:
            if chunk:
                response_chunks.append(chunk)
        
        # Combine chunks into final response
        response_text = "".join(response_chunks) if response_chunks else ""
        
        # Add assistant response to chat history
        assistant_msg = Message(content=response_text, role="assistant")
        chat_history.append(assistant_msg)
        
        return response_text

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        deployment_name: str | None = None,
        metrics: Optional[List[Type[EvaluationMetric]]] = None,
        scenarios_file: Optional[Path] = None,
        api_version: Optional[str] = None,
        results_dir: Optional[Path] = None
    ):
        """Initialize the evaluator with configuration."""
        # Set up the evaluator
        self.evaluator = metrics[0] if metrics else BrandVoiceMetric()
        
        # Validate credentials
        if not all([endpoint, api_key, deployment_name]):
            raise ValueError("All Azure OpenAI credentials (endpoint, api_key, deployment_name) are required")
            
        self.endpoint = str(endpoint)
        self.api_key = str(api_key)
        self.deployment_name = str(deployment_name)
        self.api_version = api_version or self.DEFAULT_API_VERSION
        
        # Initialize Azure OpenAI client via shared factory
        self.client = create_azure_openai_client(
            api_key=self.api_key,
            endpoint=self.endpoint,
            api_version=self.api_version,
        )
        
        # Set up metrics
        self.metrics = self._initialize_metrics(metrics)
        
        # Load scenarios
        self.scenarios = self._load_scenarios(scenarios_file)
        
        # Initialize flows
        self._initialize_flows()
        
        # Set up results directory
        self.results_dir = results_dir or Path(self.DEFAULT_RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _initialize_flows(self, brand_name: Optional[str] = None) -> None:
        """Initialize conversation flows with Azure OpenAI credentials and brand configuration."""
        self.flows = {
            "preamble": PreambleFlowSK(
                ENDPOINT=self.endpoint,
                API_KEY=self.api_key,
                DEPLOYMENT_NAME=self.deployment_name,
                BRAND_NAME=brand_name
            ),
            "order": OrderAssistantFlowSK(
                ENDPOINT=self.endpoint,
                API_KEY=self.api_key,
                DEPLOYMENT_NAME=self.deployment_name,
                BRAND_NAME=brand_name
            ),
            "summary": SummaryFlowSK(
                ENDPOINT=self.endpoint,
                API_KEY=self.api_key,
                DEPLOYMENT_NAME=self.deployment_name,
                BRAND_NAME=brand_name
            )
        }
        
    def _initialize_metrics(self, metric_classes: Optional[List[Type[EvaluationMetric]]] = None) -> List[EvaluationMetric]:
        """Initialize evaluation metrics."""
        if metric_classes is None:
            metric_classes = [BrandVoiceMetric, RelevanceMetric, TaskCompletionMetric]
            
        return [metric_class() for metric_class in metric_classes]
        
    def _load_scenarios(self, scenarios_file: Optional[Path] = None) -> List[ScenarioConfig]:
        """Load test scenarios from file or use defaults."""
        if scenarios_file and scenarios_file.exists():
            return ScenarioLoader.load_scenarios(scenarios_file)
            
        # Default scenarios if no scenario  for user prompt file is not available
        return [ScenarioConfig(text=text, type="general") 
                for text in ["Hello",
                "I want to order dinner for my family",
                "Can you put together a meal for 4 people?",
                "I will need a glutten-free option",
                "Do you have a desert option?",
                "Include medium size drinks for everyone",
                "Increase the order by 2 people"]
        ]
        
          
    async def simulate_all_scenarios(self, brand_config: Dict[str, Any], save_results: bool = True) -> List[EvaluationResult]:
        """Run evaluation across all scenarios following the complete conversation flow."""
        results = []
        conversation = []
        chat_history = []
        current_order = {"items": []}
        
        try:
            # Initialize flows with brand configuration
            self._initialize_flows(brand_name=brand_config["name"])
            
            # preamble flow - greeting
            greeting_response = await self._handle_conversation_turn(
                user_message="Hello", #starting user prompt
                chat_history=chat_history,
                current_order=current_order,
                flow_type="preamble"
            )
            conversation.append({"role": "user", "content": "Hello"})
            conversation.append({"role": "assistant", "content": greeting_response})
            
            # Process each order scenario
            for scenario in self.scenarios:
                response_text = await self._handle_conversation_turn(
                    user_message=scenario.text,
                    chat_history=chat_history,
                    current_order=current_order,
                    flow_type="order"
                )
                conversation.append({"role": "user", "content": scenario.text})
                conversation.append({"role": "assistant", "content": response_text})
            
            # end with summary flow
            summary_response = await self._handle_conversation_turn(
                user_message="Can you summarize my order?", #final user prompt
                chat_history=chat_history,
                current_order=current_order,
                flow_type="summary"
            )
            conversation.append({"role": "user", "content": "Can you summarize my order?"})
            conversation.append({"role": "assistant", "content": summary_response})
            
            # Evaluating the full conversation
            scores = {}
            errors = []
            
            for metric in self.metrics:
                try:
                    from streaming_ordering_chatbot.api.utils.azure_client import build_chat_params
                    params = build_chat_params({"temperature": 0.7, "max_tokens": 500})
                    response = self.client.chat.completions.create(
                        model=self.deployment_name,
                        messages=[
                            {
                                "role": "system", 
                                "content": metric.get_system_prompt(brand_config)
                            },
                            {
                                "role": "user", 
                                "content": metric.format_conversation(conversation)
                            }
                        ],
                        **params,
                    )
                    
                    content = response.choices[0].message.content
                    if content is None:
                        raise ValueError("Received empty response from OpenAI API")
                    score = metric.parse_score(content)
                    scores[metric.__class__.__name__] = score
                    
                except Exception as e:
                    self.logger.error(f"Error evaluating metric {metric.__class__.__name__}: {str(e)}")
                    errors.append(f"Metric {metric.__class__.__name__} failed: {str(e)}")
                    continue
            
            status = (
                EvaluationStatus.SUCCESS if not errors
                else EvaluationStatus.PARTIAL_SUCCESS if scores
                else EvaluationStatus.FAILURE
            )
            
            result = EvaluationResult(
                scenario_id="full_conversation",
                brand_name=brand_config["name"],
                scores=scores,
                conversation=conversation,
                timestamp=datetime.now(),
                status=status,
                errors=errors if errors else None
            )
            
            results.append(result)
            
            if save_results:
                self._save_result(result)
            
        except Exception as e:
            self.logger.error(
                f"Failed conversation: {str(e)}"
            )
                    
        return results
        
    def _save_result(self, result: EvaluationResult) -> None:
        timestamp = result.timestamp.strftime("%Y%m%d_%H%M%S") # to support tracing
        filename = f"{result.brand_name}_conversation_{timestamp}.json"
        
        try:
            filepath = self.results_dir / filename
            with open(filepath, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            self.logger.info(f"Saved evaluation result to {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to save result: {str(e)}")
            
    def _validate_brand_config(self, config: Dict[str, Any]) -> None:
        """Validate brand configuration."""
        missing_fields = [
            field for field in self.REQUIRED_BRAND_FIELDS
            if field not in config
        ]
        
        if missing_fields:
            raise BrandConfigValidationError(
                f"Missing required fields in brand config: {', '.join(missing_fields)}"
            )

    async def run_evaluation(self, brand_configs: Dict[str, Dict]) -> Dict:
        """Run the complete evaluation across all brands."""
        all_results = []
        
        # Validate all brand configs before starting
        for brand_name, brand_config in brand_configs.items():
            self._validate_brand_config(brand_config)
        
        # Run evaluations for each brand
        for brand_name, brand_config in brand_configs.items():
            self.logger.info(f"\nEvaluating brand: {brand_config['name']}")
            try:
                results = await self.simulate_all_scenarios(brand_config)
                all_results.extend(results)
            except Exception as e:
                self.logger.error(f"Error evaluating brand {brand_name}: {e}")
                continue
        
        # Format results for saving
        formatted_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_brands": len(brand_configs),
                "num_scenarios": len(self.scenarios)
            },
            "evaluation": [result.to_dict() for result in all_results]
        }
        
        return formatted_results
    
    async def save_results(self, results: Dict, output_dir: Optional[Path] = None):
        """Save evaluation results to files."""
        if output_dir is None:
            output_dir = Path.cwd() / "evaluation_results"
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Save raw results
        with open(output_dir / "chatbot_evaluation_data.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Save evaluation score and conversation results
        with open(output_dir / "evaluation_metrics.json", "w") as f:
            json.dump(results["evaluation"], f, indent=2)
        
        print(f"\nEvaluation complete! Results saved to {output_dir}")
        print(f"Tested {results['metadata']['num_brands']} brands")
        print(f"Evaluated {results['metadata']['num_scenarios']} scenarios per brand")

async def main():
    """Main function to run the evaluation."""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get Azure OpenAI configuration
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([endpoint, api_key, deployment_name]):
        raise ValueError("Missing required environment variables")
    
    # Load brand configurations
    brand_config_path = Path("streaming_ordering_chatbot/resources/brand_configs.json")
    if not brand_config_path.exists():
        raise FileNotFoundError(f"Brand configuration file not found: {brand_config_path}")
        
    with open(brand_config_path, "r") as f:
        brand_configs = json.load(f)
    
    # Load scenarios
    scenarios_path = Path("streaming_ordering_chatbot/resources/evaluation_scenarios.json")
    if not scenarios_path.exists():
        print(f"Warning: No scenarios file found at {scenarios_path}, using defaults")
    
    # Initialize evaluator
    evaluator = BrandedChatbotEvaluator(
        endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment_name,
        scenarios_file=scenarios_path if scenarios_path.exists() else None
    )
    
    try:
        all_results = []
        # Run evaluation for each brand
        for brand_name, brand_config in brand_configs.items():
            print(f"\nEvaluating brand: {brand_config['name']}")
            brand_results = await evaluator.simulate_all_scenarios(brand_config)
            all_results.extend(brand_results)
        
        # Format results for saving
        formatted_results = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "num_brands": len(brand_configs),
                "num_scenarios": len(evaluator.scenarios)
            },
            "evaluation": [result.to_dict() for result in all_results]
        }
        
        # Save results
        await evaluator.save_results(formatted_results)
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
