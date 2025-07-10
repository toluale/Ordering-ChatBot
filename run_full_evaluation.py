#!/usr/bin/env python3
# run_full_evaluation.py

"""
Complete conversation generation and evaluation pipeline.
This script runs both the conversation generator and evaluator in sequence.
"""

import asyncio
import sys
import os
from pathlib import Path
import logging

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from conversation_generator import ConversationGenerator
from conversation_evaluator import ConversationEvaluator


def validate_environment():
    """Validate required environment variables."""
    required_vars = [
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_API_KEY", 
        "AZURE_OPENAI_DEPLOYMENT_NAME"
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
    
    return {
        "endpoint": str(os.getenv("AZURE_OPENAI_ENDPOINT")),
        "api_key": str(os.getenv("AZURE_OPENAI_API_KEY")),
        "deployment_name": str(os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"))
    }


def load_brand_configs():
    """Load brand configurations."""
    brand_config_path = Path("streaming_ordering_chatbot/resources/brand_configs.json")
    if not brand_config_path.exists():
        raise FileNotFoundError(f"Brand configuration file not found: {brand_config_path}")
        
    import json
    with open(brand_config_path, "r") as f:
        return json.load(f)


async def run_generation_phase(azure_config, brand_configs):
    """Run the conversation generation phase."""
    print("Starting Conversation Generation Phase...")
    print("=" * 50)
    
    # Load scenarios
    scenarios_path = Path("streaming_ordering_chatbot/resources/evaluation_scenarios.json")
    if not scenarios_path.exists():
        print(f"Warning: No scenarios file found at {scenarios_path}, using defaults")
    
    # Initialize conversation generator
    generator = ConversationGenerator(
        endpoint=azure_config["endpoint"],
        api_key=azure_config["api_key"],
        deployment_name=azure_config["deployment_name"],
        scenarios_file=scenarios_path if scenarios_path.exists() else None
    )
    
    # Generate conversations for all brands
    generated_files = await generator.generate_all_conversations(brand_configs)
    
    print(f"\nConversation Generation Complete!")
    print(f"Generated {len(generated_files)} conversation files:")
    for filepath in generated_files:
        print(f" {filepath.name}")
    
    return generated_files


async def run_evaluation_phase(azure_config):
    """Run the conversation evaluation phase."""
    print("\n Starting Conversation Evaluation Phase...")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = ConversationEvaluator(
        endpoint=azure_config["endpoint"],
        api_key=azure_config["api_key"],
        deployment_name=azure_config["deployment_name"]
    )
    
    # Evaluate all conversations
    results = await evaluator.evaluate_all_conversations()
    
    if not results:
        print("No evaluation results generated")
        return None
    
    # Save detailed results to CSV
    csv_file = evaluator.save_results_to_csv(results)
    print(f"Detailed results saved to: {csv_file.name}")
    
    # Generate and save brand summary
    summary = evaluator.generate_brand_summary(results)
    summary_file = evaluator.save_brand_summary(summary)
    print(f" Brand summary saved to: {summary_file.name}")
    
    # Print summary to console
    print(f"\n Evaluation Complete!")
    print(f"Summary:")
    print(f"  Total conversations evaluated: {len(set(r.conversation_file for r in results))}")
    print(f"  Total brands: {len(set(r.brand_name for r in results))}")
    print(f"  Total evaluations: {len(results)}")
    
    print(f"\n Brand Performance Overview:")
    for brand, data in summary.items():
        print(f"\n  {brand}:")
        print(f"    Overall Average Score: {data['overall_average']:.3f}")
        print(f"    Conversations Evaluated: {data['total_conversations']}")
        for metric, metric_data in data['metrics'].items():
            avg_score = metric_data['average_score']
            std_score = metric_data['std_score']
            print(f"    {metric}: {avg_score:.3f} (±{std_score:.3f})")
    
    return results, summary


async def main():
    """Main function to run the complete evaluation pipeline."""
    try:
        # Setup logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        print("Starting Complete Conversation Evaluation Pipeline")
        print("=" * 60)
        
        # Validate environment
        azure_config = validate_environment()
        print("Environment variables validated")
        
        # Load brand configurations
        brand_configs = load_brand_configs()
        print(f"Loaded {len(brand_configs)} brand configurations")
        
        # Phase 1: Generate Conversations
        generated_files = await run_generation_phase(azure_config, brand_configs)
        
        if not generated_files:
            print("No conversations generated, stopping pipeline")
            return
        
        # Phase 2: Evaluate Conversations
        evaluation_results = await run_evaluation_phase(azure_config)
        
        if evaluation_results is None:
            print(" Evaluation failed, but conversations were generated successfully")
            return
        
        results, summary = evaluation_results
        
        print(f"\nPipeline Complete!")
        print(f"Check the 'evaluation_results' directory for:")
        print(f"  Generated conversation files")
        print(f"  Detailed evaluation CSV")
        print(f"  Brand performance summary JSON")
        
    except KeyboardInterrupt:
        print("\n Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nPipeline failed with error: {e}")
        logging.exception("Pipeline error")
        sys.exit(1)


if __name__ == "__main__":
    # Load environment variables
    from dotenv import load_dotenv
    load_dotenv()
    
    asyncio.run(main())
