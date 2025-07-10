# conversation_evaluator.py

import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Type
import json
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass
import pandas as pd
import re

from openai import AzureOpenAI
from streaming_ordering_chatbot.evaluation.metrics import (
    EvaluationMetric, BrandVoiceMetric, RelevanceMetric, TaskCompletionMetric
)


@dataclass
class EvaluationResult:
    """Results of a conversation evaluation."""
    brand_name: str
    conversation_file: str
    metric_name: str
    score: float
    explanation: str
    timestamp: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            "brand_name": self.brand_name,
            "conversation_file": self.conversation_file,
            "metric_name": self.metric_name,
            "score": self.score,
            "explanation": self.explanation,
            "timestamp": self.timestamp.isoformat()
        }


class ConversationEvaluator:
    """Evaluates pre-generated conversations using various metrics."""
    
    DEFAULT_API_VERSION = "2024-12-01-preview"
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment_name: str,
        metrics: Optional[List[Type[EvaluationMetric]]] = None,
        api_version: Optional[str] = None,
        conversations_dir: Optional[Path] = None
    ):
        """Initialize the evaluator."""
        self.endpoint = endpoint
        self.api_key = api_key
        self.deployment_name = deployment_name
        self.api_version = api_version or self.DEFAULT_API_VERSION
        
        # Initialize Azure OpenAI client
        self.client = AzureOpenAI(
            azure_endpoint=self.endpoint,
            api_key=self.api_key,
            api_version=self.api_version
        )
        
        # Set up metrics
        self.metrics = self._initialize_metrics(metrics)
        
        # Set up conversations directory
        self.conversations_dir = conversations_dir or Path("evaluation_results")
        if not self.conversations_dir.exists():
            raise FileNotFoundError(f"Conversations directory not found: {self.conversations_dir}")
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _initialize_metrics(self, metric_classes: Optional[List[Type[EvaluationMetric]]] = None) -> List[EvaluationMetric]:
        """Initialize evaluation metrics."""
        if metric_classes is None:
            metric_classes = [BrandVoiceMetric, RelevanceMetric, TaskCompletionMetric]
            
        return [metric_class() for metric_class in metric_classes]
    
    def load_conversation_files(self) -> List[Path]:
        """Load all conversation JSON files from the directory."""
        pattern = "*_conversation_*.json"
        conversation_files = list(self.conversations_dir.glob(pattern))
        
        if not conversation_files:
            raise FileNotFoundError(f"No conversation files found in {self.conversations_dir}")
        
        self.logger.info(f"Found {len(conversation_files)} conversation files")
        return conversation_files
    
    def load_conversation_data(self, filepath: Path) -> Dict[str, Any]:
        """Load conversation data from JSON file."""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading conversation file {filepath}: {e}")
            raise
    
    def load_brand_configs(self) -> Dict[str, Dict[str, Any]]:
        """Load brand configurations."""
        brand_config_path = Path("streaming_ordering_chatbot/resources/brand_configs.json")
        if not brand_config_path.exists():
            raise FileNotFoundError(f"Brand configuration file not found: {brand_config_path}")
            
        with open(brand_config_path, "r") as f:
            return json.load(f)
    
    def extract_explanation_from_response(self, response: str) -> str:
        """Extract explanation from the metric evaluation response."""
        # Try to find explanation after the score
        lines = response.strip().split('\n')
        explanation_lines = []
        found_score = False
        
        for line in lines:
            # Skip the line with just the score
            if re.match(r'^0?\.[0-9]+$', line.strip()):
                found_score = True
                continue
            
            # Collect explanation lines
            if line.strip():
                explanation_lines.append(line.strip())
        
        explanation = ' '.join(explanation_lines)
        
        # If no clear explanation found, use the full response minus the score
        if not explanation or len(explanation) < 10:
            # Remove numeric scores from response
            explanation = re.sub(r'\b0?\.[0-9]+\b', '', response)
            explanation = ' '.join(explanation.split())
        
        return explanation[:500] if explanation else "No explanation provided"
    
    async def evaluate_conversation(
        self, 
        conversation_data: Dict[str, Any], 
        brand_config: Dict[str, Any],
        conversation_file: str
    ) -> List[EvaluationResult]:
        """Evaluate a single conversation using all metrics."""
        results = []
        conversation = conversation_data.get("conversation", [])
        brand_name = conversation_data.get("brand_name", "Unknown")
        
        self.logger.info(f"Evaluating conversation for brand: {brand_name}")
        
        for metric in self.metrics:
            try:
                self.logger.info(f"Applying metric: {metric.__class__.__name__}")
                
                # Get evaluation from OpenAI
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
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=500
                )
                
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("Received empty response from OpenAI API")
                
                # Parse score and extract explanation
                score = metric.parse_score(content)
                explanation = self.extract_explanation_from_response(content)
                
                result = EvaluationResult(
                    brand_name=brand_name,
                    conversation_file=conversation_file,
                    metric_name=metric.__class__.__name__,
                    score=score,
                    explanation=explanation,
                    timestamp=datetime.now()
                )
                
                results.append(result)
                self.logger.info(f"Score for {metric.__class__.__name__}: {score:.3f}")
                
            except Exception as e:
                self.logger.error(f"Error evaluating metric {metric.__class__.__name__}: {str(e)}")
                # Add a failed result
                failed_result = EvaluationResult(
                    brand_name=brand_name,
                    conversation_file=conversation_file,
                    metric_name=metric.__class__.__name__,
                    score=0.0,
                    explanation=f"Evaluation failed: {str(e)}",
                    timestamp=datetime.now()
                )
                results.append(failed_result)
                continue
        
        return results
    
    async def evaluate_all_conversations(self) -> List[EvaluationResult]:
        """Evaluate all conversation files."""
        conversation_files = self.load_conversation_files()
        brand_configs = self.load_brand_configs()
        all_results = []
        
        for filepath in conversation_files:
            try:
                # Load conversation data
                conversation_data = self.load_conversation_data(filepath)
                brand_name = conversation_data.get("brand_name", "Unknown")
                
                # Find matching brand config
                brand_config = None
                for config_key, config_data in brand_configs.items():
                    if config_data.get("name") == brand_name:
                        brand_config = config_data
                        break
                
                if not brand_config:
                    self.logger.warning(f"No brand config found for {brand_name}, skipping")
                    continue
                
                # Evaluate conversation
                results = await self.evaluate_conversation(
                    conversation_data, 
                    brand_config, 
                    filepath.name
                )
                all_results.extend(results)
                
            except Exception as e:
                self.logger.error(f"Error processing file {filepath}: {e}")
                continue
        
        return all_results
    
    def save_results_to_csv(self, results: List[EvaluationResult], output_file: Optional[Path] = None) -> Path:
        """Save evaluation results to CSV file grouped by brand."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.conversations_dir / f"evaluation_scores_{timestamp}.csv"
        
        # Convert results to DataFrame
        data = []
        for result in results:
            data.append({
                'Brand': result.brand_name,
                'Conversation_File': result.conversation_file,
                'Metric': result.metric_name,
                'Score': result.score,
                'Explanation': result.explanation,
                'Timestamp': result.timestamp.isoformat()
            })
        
        df = pd.DataFrame(data)
        
        # Sort by brand and metric for better organization
        df = df.sort_values(['Brand', 'Metric', 'Conversation_File'])
        
        # Save to CSV
        df.to_csv(output_file, index=False, encoding='utf-8')
        self.logger.info(f"Evaluation results saved to {output_file}")
        
        return output_file
    
    def generate_brand_summary(self, results: List[EvaluationResult]) -> Dict[str, Dict[str, Any]]:
        """Generate summary statistics grouped by brand."""
        df = pd.DataFrame([result.to_dict() for result in results])
        
        summary = {}
        for brand in df['brand_name'].unique():
            brand_data = df[df['brand_name'] == brand]
            
            brand_summary = {
                'total_conversations': brand_data['conversation_file'].nunique(),
                'metrics': {}
            }
            
            for metric in brand_data['metric_name'].unique():
                metric_data = brand_data[brand_data['metric_name'] == metric]
                
                brand_summary['metrics'][metric] = {
                    'average_score': float(metric_data['score'].mean()),
                    'min_score': float(metric_data['score'].min()),
                    'max_score': float(metric_data['score'].max()),
                    'std_score': float(metric_data['score'].std()),
                    'explanations': metric_data['explanation'].tolist()
                }
            
            # Calculate overall average
            brand_summary['overall_average'] = float(brand_data['score'].mean())
            
            summary[brand] = brand_summary
        
        return summary
    
    def save_brand_summary(self, summary: Dict[str, Dict[str, Any]], output_file: Optional[Path] = None) -> Path:
        """Save brand summary to JSON file."""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = self.conversations_dir / f"brand_evaluation_summary_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Brand summary saved to {output_file}")
        return output_file


async def main():
    """Main function to evaluate conversations."""
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    # Get Azure OpenAI configuration
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([endpoint, api_key, deployment_name]):
        raise ValueError("Missing required environment variables")
    
    # Type assertions since we've validated they're not None
    endpoint = str(endpoint)
    api_key = str(api_key)
    deployment_name = str(deployment_name)
    
    # Initialize evaluator
    evaluator = ConversationEvaluator(
        endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment_name
    )
    
    try:
        # Evaluate all conversations
        print("Starting conversation evaluation...")
        results = await evaluator.evaluate_all_conversations()
        
        if not results:
            print("No evaluation results generated")
            return
        
        # Save detailed results to CSV
        csv_file = evaluator.save_results_to_csv(results)
        print(f"Detailed results saved to: {csv_file}")
        
        # Generate and save brand summary
        summary = evaluator.generate_brand_summary(results)
        summary_file = evaluator.save_brand_summary(summary)
        print(f"Brand summary saved to: {summary_file}")
        
        # Print summary to console
        print(f"\nEvaluation Summary:")
        print(f"Total conversations evaluated: {len(set(r.conversation_file for r in results))}")
        print(f"Total brands: {len(set(r.brand_name for r in results))}")
        print(f"Total evaluations: {len(results)}")
        
        print(f"\nBrand Performance Overview:")
        for brand, data in summary.items():
            print(f"\n{brand}:")
            print(f"  Overall Average Score: {data['overall_average']:.3f}")
            print(f"  Conversations Evaluated: {data['total_conversations']}")
            for metric, metric_data in data['metrics'].items():
                print(f"  {metric}: {metric_data['average_score']:.3f} (±{metric_data['std_score']:.3f})")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
