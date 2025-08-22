import os
import json
import logging
from typing import Dict, Any
from semantic_kernel.functions.kernel_function_decorator import kernel_function

logger = logging.getLogger(__name__)


class BrandPlugin:    
    def __init__(self, brand_config_path: str = "brand_configs.json"):
        self.config_path = brand_config_path
        self.brand_configs = self._load_brand_configs()
        self.current_brand = os.getenv("RESTAURANT_BRAND", "contoso")
    
    def _load_brand_configs(self) -> Dict[str, Any]:
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading brand configs: {e}")
            return {}
    
    @kernel_function(description="Get brand personality instructions", name="get_brand_personality")
    def get_brand_personality(self) -> str:
        if not self.current_brand or self.current_brand not in self.brand_configs:
            return ""

        config = self.brand_configs[self.current_brand]
        personality = f"""BRAND PERSONALITY FOR {config['name']}:
                        - Tone: {config['tone']}
                        - Style: {config['style']}
                        - Key Phrases: {', '.join(config['key_phrases'])}
                        - Values: {', '.join(config['values'])}

                        Apply this personality consistently in all responses while maintaining professionalism.
                        """
        return personality

    @kernel_function(description="Get current brand name", name="get_brand_name")
    def get_brand_name(self) -> str:
        if self.current_brand and self.current_brand in self.brand_configs:
            return self.brand_configs[self.current_brand]['name']
        return "Restaurant"
    
    @kernel_function(description="Set current brand", name="set_brand")
    def set_brand(self, brand_key: str) -> str:
        if brand_key in self.brand_configs:
            self.current_brand = brand_key
            return f"Brand set to {self.brand_configs[brand_key]['name']}"
        return f"Brand '{brand_key}' not found in configuration"
    
    @kernel_function(description="List available brands", name="list_brands")
    def list_brands(self) -> str:
        if not self.brand_configs:
            return "No brand configurations available"
        
        brands = [f"- {key}: {config['name']}" for key, config in self.brand_configs.items()]
        return "Available brands:\n" + "\n".join(brands)
    
    def get_current_brand_key(self) -> str:
        return self.current_brand
    
    def get_brand_config(self, brand_key: str = None) -> Dict[str, Any]:
        key = brand_key or self.current_brand
        return self.brand_configs.get(key, {})