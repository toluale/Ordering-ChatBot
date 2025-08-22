import json
from pathlib import Path
from typing import Dict, Optional

from semantic_kernel import Kernel
from semantic_kernel.functions.kernel_function_decorator import kernel_function
from semantic_kernel.functions.kernel_arguments import KernelArguments
from .menu_manager import get_menu_manager


class BrandPersonalityPlugin:
    """For managing brand personality in conversations."""

    def __init__(self, kernel: Kernel, brand_name: Optional[str] = None):
        """Initialize the brand personality plugin."""
        self.kernel = kernel
        self._load_brand_configs()
        self.menu_manager = get_menu_manager()
        # Prefer explicit brand_name if valid; else follow MenuManager's current brand
        if brand_name and brand_name in self.brand_configs:
            self.current_brand = brand_name
        else:
            self.current_brand = self.menu_manager.get_current_brand()

    def _load_brand_configs(self):
        """Load brand configurations from JSON file."""
        config_path = Path(__file__).parent.parent.parent.joinpath("resources/brand_configs.json")
        try:
            with open(config_path, "r") as f:
                self.brand_configs = json.load(f)
        except Exception as e:
            print(f"Error loading brand configs: {e}")
            self.brand_configs = {}

    def set_brand(self, brand_name: str) -> bool:
        """Set the current brand personality.
        Returns:
            bool: True if brand was successfully set, False if brand not found
        """
        if brand_name in self.brand_configs:
            self.current_brand = brand_name
            # Keep MenuManager in sync to avoid divergent state
            try:
                self.menu_manager.set_current_brand(brand_name)
            except Exception:
                pass
            return True
        return False   
     
    @kernel_function(description="Get brand personality instructions", name="get_brand_instructions")
    def get_brand_instructions(self, brand_name: Optional[str] = None) -> str:
        """Get the personality instructions for a specific brand."""
        # Resolve brand: prefer provided, then MenuManager, then local
        brand_name = brand_name or self.menu_manager.get_current_brand() or self.current_brand
        if not brand_name:
            return "No brand personality selected."
            
        brand = self.brand_configs.get(brand_name, {})
        if not brand:
            return f"Brand '{brand_name}' not found."
        
        return f"""You are representing {brand['name']}.

                TONE AND STYLE:
                - Tone: {brand['tone']}
                - Style: {brand['style']}

                BRAND VOICE GUIDELINES:
                1. Key phrases to naturally incorporate:
                {', '.join(brand['key_phrases'])}

                2. Core brand values to embody:
                {', '.join(brand['values'])}

                Remember to maintain this brand voice consistently throughout the conversation while remaining helpful and natural."""

    @kernel_function(description="Apply brand personality to system prompt", name="enhance_system_prompt")
    def enhance_system_prompt(self, system_prompt: str) -> str:
        """Enhance a system prompt with brand personality instructions."""
        if not self.current_brand:
            return system_prompt
            
        brand_instructions = self.get_brand_instructions()
        return f"{brand_instructions}\n\nBASE INSTRUCTIONS:\n{system_prompt}"

    @kernel_function(description="Format message with brand personality", name="format_brand_message")
    def format_brand_message(self, message: str, brand_name: Optional[str] = None) -> str:
        """Format a message according to brand personality."""
        brand_name = brand_name or self.current_brand
        if not brand_name:
            return message
            
        brand = self.brand_configs.get(brand_name, {})
        if not brand:
            return message

        # Add brand-specific context
        context = self.get_brand_instructions(brand_name)
        return f"{context}\n\nResponse in this style: {message}"

    #@kernel_function(description="List available brand personalities", name="list_brands")
    def list_brands(self) -> str:
        """List all available brand personalities with their key characteristics."""
        if not self.brand_configs:
            return "No brand configurations available."
            
        brands_info = []
        for brand_name, config in self.brand_configs.items():
            brands_info.append(
                f"- {config['name']}:\n"
                f"  Tone: {config['tone']}\n"
                f"  Values: {', '.join(config['values'])}"
            )
        return "\n\n".join(brands_info)

    @kernel_function(description="Get the current brand personality", name="get_current_brand")
    def get_current_brand(self) -> str:
        """Get information about the currently selected brand (detailed)."""
        # Prefer MenuManager's brand if set
        brand_name = self.menu_manager.get_current_brand() or self.current_brand
        if not brand_name:
            return "No brand personality currently selected."
        brand = self.brand_configs.get(brand_name, {})
        if not brand:
            return f"Error: Selected brand '{brand_name}' not found in configurations."
        return f"Current brand: {brand['name']}\nTone: {brand['tone']}\nStyle: {brand['style']}"

    @kernel_function(description="Get the current brand name only", name="get_current_brand_name")
    def get_current_brand_name(self) -> str:
        """Return only the brand name string, or empty if none set."""
        brand_name = self.menu_manager.get_current_brand() or self.current_brand
        return brand_name or ""
    

    '''
    @kernel_function(description="Apply brand personality with conversation style", name="enhance_with_style")
    def enhance_with_style(self, system_prompt: str, conversation_style: Optional[str] = None) -> str:
        """Enhance a system prompt with both brand personality and conversation style."""
        if not self.current_brand:
            return system_prompt
        
        # Get base brand instructions
        brand_instructions = self.get_brand_instructions()
        
        # Get style instructions from style plugin if available
        style_instructions = ""
        if hasattr(self, 'kernel'):
            try:
                style_result = self.kernel.invoke(
                    plugin_name="style",
                    function_name="get_style_instructions",
                    arguments=KernelArguments(style=conversation_style) if conversation_style else None
                )
                style_instructions = str(style_result) if style_result else ""
            except:
                pass  # Style plugin not available or error occurred
        
        # Combine brand + style + system prompt
        enhanced_instructions = brand_instructions
        if style_instructions:
            enhanced_instructions = f"{brand_instructions}\n\nCONVERSATION STYLE:\n{style_instructions}"
        
        return f"{enhanced_instructions}\n\nBASE INSTRUCTIONS:\n{system_prompt}"
    '''