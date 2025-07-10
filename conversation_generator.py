# conversation_generator.py

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import json
import asyncio
import logging
from datetime import datetime
from dataclasses import dataclass

from streaming_ordering_chatbot.api.flows import (PreambleFlowSK, OrderAssistantFlowSK, SummaryFlowSK)
from streaming_ordering_chatbot.api.models import Message
from streaming_ordering_chatbot.evaluation.scenarios import (ScenarioLoader, ScenarioConfig)


@dataclass
class ConversationData:
    """Data structure for a complete conversation."""
    brand_name: str
    conversation: List[Dict[str, str]]
    timestamp: datetime
    scenarios: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert conversation data to dictionary for serialization."""
        return {
            "brand_name": self.brand_name,
            "conversation": self.conversation,
            "timestamp": self.timestamp.isoformat(),
            "scenarios": self.scenarios,
            "metadata": {
                "total_turns": len(self.conversation),
                "user_turns": len([turn for turn in self.conversation if turn["role"] == "user"]),
                "assistant_turns": len([turn for turn in self.conversation if turn["role"] == "assistant"])
            }
        }


class ConversationGenerator:
    """Generates conversations for different restaurant brands."""
    
    DEFAULT_RESULTS_DIR = "evaluation_results"
    
    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment_name: str,
        scenarios_file: Optional[Path] = None,
        results_dir: Optional[Path] = None
    ):
        """Initialize the conversation generator."""
        if not all([endpoint, api_key, deployment_name]):
            raise ValueError("All Azure OpenAI credentials (endpoint, api_key, deployment_name) are required")
            
        self.endpoint = str(endpoint)
        self.api_key = str(api_key)
        self.deployment_name = str(deployment_name)
        
        # Load scenarios
        self.scenarios = self._load_scenarios(scenarios_file)
        
        # Set up results directory
        self.results_dir = results_dir or Path(self.DEFAULT_RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def _load_scenarios(self, scenarios_file: Optional[Path] = None) -> List[ScenarioConfig]:
        """Load test scenarios from file or use defaults."""
        if scenarios_file and scenarios_file.exists():
            return ScenarioLoader.load_scenarios(scenarios_file)
            
        # Default scenarios if no scenario file is available
        print(f"Using default scenarios")
        return [ScenarioConfig(text=text, type="general") 
                for text in [
                    "I want to order dinner for my family",
                    "Can you put together a meal for 4 people?",
                    "I will need a gluten-free option",
                    "Do you have a dessert option?",
                    "Include medium size drinks for everyone",
                    "Increase the order by 2 people"
                ]]
        
    def _initialize_flows(self, brand_name: str, conversation_style: Optional[str] = None) -> Dict[str, Any]:
        """Initialize conversation flows with Azure OpenAI credentials and brand configuration.
        
        Args:
            brand_name: Name of the restaurant brand
            conversation_style: Optional conversation style (casual, genz, etc.). 
                              If None, defaults to the brand's original style.
        """
        return {
            "preamble": PreambleFlowSK(
                endpoint=self.endpoint,
                api_key=self.api_key,
                deployment_name=self.deployment_name,
                brand_name=brand_name,
                conversation_style=conversation_style
            ),
            "order": OrderAssistantFlowSK(
                endpoint=self.endpoint,
                api_key=self.api_key,
                deployment_name=self.deployment_name,
                brand_name=brand_name,
                conversation_style=conversation_style
            ),
            "summary": SummaryFlowSK(
                endpoint=self.endpoint,
                api_key=self.api_key,
                deployment_name=self.deployment_name,
                brand_name=brand_name,
                conversation_style=conversation_style
            )
        }
        
    async def _handle_conversation_turn(
        self,
        user_message: str,
        chat_history: List[Message],
        current_order: Dict[str, Any],
        flows: Dict[str, Any],
        flow_type: str = "order"
    ) -> str:
        """Handle a single turn in the conversation using the specified flow."""
        # Add user message to chat history
        user_msg = Message(content=user_message, role="user")
        chat_history.append(user_msg)
        
        # Get assistant response using direct call to appropriate SK flow
        response_generator = flows[flow_type](chat_history=chat_history, current_order=current_order)
        
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
        
    async def generate_conversation(self, brand_name: str) -> ConversationData:
        """Generate a complete conversation for a specific brand."""
        conversation = []
        chat_history = []
        current_order = {"items": []}
        scenario_texts = []
        
        try:
            # Initialize flows with brand configuration
            flows = self._initialize_flows(brand_name)
            
            # Preamble flow - greeting
            self.logger.info(f"Starting conversation for brand: {brand_name}")
            greeting_response = await self._handle_conversation_turn(
                user_message="Hello",
                chat_history=chat_history,
                current_order=current_order,
                flows=flows,
                flow_type="preamble"
            )
            conversation.append({"role": "user", "content": "Hello"})
            conversation.append({"role": "assistant", "content": greeting_response})
            
            # Process each order scenario
            for scenario in self.scenarios:
                self.logger.info(f"Processing scenario: {scenario.text[:50]}...")
                response_text = await self._handle_conversation_turn(
                    user_message=scenario.text,
                    chat_history=chat_history,
                    current_order=current_order,
                    flows=flows,
                    flow_type="order"
                )
                conversation.append({"role": "user", "content": scenario.text})
                conversation.append({"role": "assistant", "content": response_text})
                scenario_texts.append(scenario.text)
            
            # End with summary flow
            summary_response = await self._handle_conversation_turn(
                user_message="Can you summarize my order?",
                chat_history=chat_history,
                current_order=current_order,
                flows=flows,
                flow_type="summary"
            )
            conversation.append({"role": "user", "content": "Can you summarize my order?"})
            conversation.append({"role": "assistant", "content": summary_response})
            
            self.logger.info(f"Conversation completed for brand: {brand_name}")
            
            return ConversationData(
                brand_name=brand_name,
                conversation=conversation,
                timestamp=datetime.now(),
                scenarios=scenario_texts
            )
            
        except Exception as e:
            self.logger.error(f"Failed to generate conversation for brand {brand_name}: {str(e)}")
            raise
    
    def save_conversation(self, conversation_data: ConversationData) -> Path:
        """Save conversation data to JSON file."""
        timestamp = conversation_data.timestamp.strftime("%Y%m%d_%H%M%S")
        filename = f"{conversation_data.brand_name}_conversation_{timestamp}.json"
        filepath = self.results_dir / filename
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(conversation_data.to_dict(), f, indent=2, ensure_ascii=False)
            self.logger.info(f"Saved conversation to {filepath}")
            return filepath
        except Exception as e:
            self.logger.error(f"Failed to save conversation: {str(e)}")
            raise
    
    async def generate_all_conversations(self, brand_configs: Dict[str, Dict[str, Any]]) -> List[Path]:
        """Generate conversations for all brands and save them."""
        generated_files = []
        
        for brand_name, brand_config in brand_configs.items():
            try:
                self.logger.info(f"Generating conversation for brand: {brand_config['name']}")
                conversation_data = await self.generate_conversation(brand_config["name"])
                filepath = self.save_conversation(conversation_data)
                generated_files.append(filepath)
            except Exception as e:
                self.logger.error(f"Error generating conversation for brand {brand_name}: {e}")
                continue
        
        self.logger.info(f"Generated {len(generated_files)} conversation files")
        return generated_files


async def main():
    """Main function to generate conversations."""
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
    
    # Initialize conversation generator
    generator = ConversationGenerator(
        endpoint=endpoint,
        api_key=api_key,
        deployment_name=deployment_name,
        scenarios_file=scenarios_path if scenarios_path.exists() else None
    )
    
    try:
        # Generate conversations for all brands
        generated_files = await generator.generate_all_conversations(brand_configs)
        
        print(f"\nConversation generation complete!")
        print(f"Generated {len(generated_files)} conversation files")
        print(f"Files saved to: {generator.results_dir}")
        
        for filepath in generated_files:
            print(f"  - {filepath.name}")
        
    except Exception as e:
        print(f"Error during conversation generation: {e}")
        raise


if __name__ == "__main__":
    asyncio.run(main())
