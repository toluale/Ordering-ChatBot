from abc import ABC, abstractmethod
from typing import Dict, Any, List

# Custom evaluation metrics for conversation quality
class EvaluationMetric(ABC):
    """Abstract base class for conversation evaluation metrics."""
    
    @abstractmethod
    def get_system_prompt(self, brand_config: Dict[str, Any]) -> str:
        """Get the system prompt for this metric.
           Returns system prompt for the evaluator
        """
        pass
        
    @abstractmethod
    def format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        """Format a conversation for evaluation.
        """
        pass
        
    @abstractmethod
    def parse_score(self, response: str) -> float:
        """Parse a score from an evaluator response.
           Returns score between 0 and 1
        """
        pass
    
    async def evaluate(self, query: str, response: str, context: dict) -> float:
        """Evaluate a response and return a score between 0 and 1."""
        raise NotImplementedError("Subclasses must implement evaluate()")


class BrandVoiceMetric(EvaluationMetric):
    """Evaluates consistency with brand voice and personality."""
    
    def get_system_prompt(self, brand_config: Dict[str, Any]) -> str:
        return f"""
        You are evaluating the brand voice consistency of a conversation.
        The brand has the following characteristics:
        - Name: {brand_config['name']}
        - Tone: {brand_config['tone']}
        - Style: {brand_config['style']}
        - Values: {', '.join(brand_config['values'])}
        
        Rate how well the assistant's responses match the brand voice from 0 to 1.
        Focus on:
        1. Tone consistency
        2. Style adherence
        3. Value alignment
        
        Provide your rating as a decimal number between 0 and 1, followed by a brief explanation.
        """
    
    def format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        return "\n".join(
            f"{turn['role'].title()}: {turn['content']}"
            for turn in conversation
        )
    
    def parse_score(self, response: str) -> float:
        # Extract first number between 0 and 1 from response
        import re
        matches = re.findall(r"0?\.[0-9]+", response)
        return float(matches[0]) if matches else 0.0


class RelevanceMetric(EvaluationMetric):
    """Evaluates response relevance and appropriateness."""
    
    def get_system_prompt(self, brand_config: Dict[str, Any]) -> str:
        return f"""
        You are evaluating the relevance and appropriateness of responses in a conversation.
        Consider the brand's:
        - Tone: {brand_config['tone']}
        - Style: {brand_config['style']}
        - Values: {', '.join(brand_config['values'])}

        Rate from 0 to 1 how well the assistant:
        1. Answers the actual question/request
        2. Provides accurate information
        3. Maintains conversation flow
        4. Gives appropriate level of detail
        
        Provide your rating as a decimal number between 0 and 1, followed by a brief explanation.
        """
    
    def format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        return "\n".join(
            f"{turn['role'].title()}: {turn['content']}"
            for turn in conversation
        )
    
    def parse_score(self, response: str) -> float:
        import re
        matches = re.findall(r"0?\.[0-9]+", response)
        return float(matches[0]) if matches else 0.0


class TaskCompletionMetric(EvaluationMetric):
    """Evaluates progress toward task completion."""
    
    def get_system_prompt(self, brand_config: Dict[str, Any]) -> str:
        return f"""
        You are evaluating how effectively the conversation progresses toward completing the ordering task.
        Consider the brand's:
        - Tone: {brand_config['tone']}
        - Style: {brand_config['style']}
        - Values: {', '.join(brand_config['values'])}

        Rate from 0 to 1 how well the assistant:
        1. Guides the order process
        2. Captures customer preferences
        3. Confirms order details
        4. Resolves issues/questions
        
        Provide your rating as a decimal number between 0 and 1, followed by a brief explanation.
        """
    
    def format_conversation(self, conversation: List[Dict[str, str]]) -> str:
        return "\n".join(
            f"{turn['role'].title()}: {turn['content']}"
            for turn in conversation
        )
    
    def parse_score(self, response: str) -> float:
        import re
        matches = re.findall(r"0?\.[0-9]+", response)
        return float(matches[0]) if matches else 0.0
