import json
from pathlib import Path
from typing import Dict, Optional

from semantic_kernel import Kernel
from semantic_kernel.functions.kernel_function_decorator import kernel_function


class BrandPersonalityPlugin:
    """Plugin for managing brand personality in conversations."""

    def __init__(self, kernel: Kernel):
        self.kernel = kernel
        self._load_brand_configs()

    def _load_brand_configs(self):
        """Load brand configurations from JSON file."""
        config_path = Path(__file__).parent.parent.parent.joinpath("resources/brand_configs.json")
        try:
            with open(config_path, "r") as f:
                self.brand_configs = json.load(f)
        except Exception as e:
            print(f"Error loading brand configs: {e}")
            self.brand_configs = {}

    @kernel_function(
        description="Get brand personality instructions",
        name="get_brand_instructions"
    )
    def get_brand_instructions(self, brand_name: str) -> str:
        """Get the personality instructions for a specific brand."""
        brand = self.brand_configs.get(brand_name, {})
        if not brand:
            return ""
        
        return f"""Brand Voice: {brand['name']}
Tone: {brand['tone']}
Style: {brand['style']}
Key phrases to incorporate: {', '.join(brand['key_phrases'])}
Brand values: {', '.join(brand['values'])}"""

    @kernel_function(
        description="Format message with brand personality",
        name="format_brand_message"
    )
    def format_brand_message(self, message: str, brand_name: str) -> str:
        """Format a message according to brand personality."""
        brand = self.brand_configs.get(brand_name, {})
        if not brand:
            return message
            
        # Add brand-specific context
        context = self.get_brand_instructions(brand_name)
        return f"{context}\n\nResponse: {message}"

    @kernel_function(
        description="List available brand personalities",
        name="list_brands"
    )
    def list_brands(self) -> str:
        """List all available brand personalities."""
        return "\n".join([
            f"- {brand}: {config['tone']}"
            for brand, config in self.brand_configs.items()
        ])
