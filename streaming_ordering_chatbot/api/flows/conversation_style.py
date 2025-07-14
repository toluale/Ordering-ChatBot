import json
from pathlib import Path
from typing import Dict, Optional
from enum import Enum

from semantic_kernel import Kernel
from semantic_kernel.functions.kernel_function_decorator import kernel_function


class ConversationStyle(Enum):
    """Available conversation styles."""
    DEFAULT = "default"
    CASUAL = "casual"
    GENZ = "genz"


class ConversationStylePlugin:
    """Plugin for managing conversation styles (casual, GenZ, default)."""

    def __init__(self, kernel: Kernel, style: ConversationStyle = ConversationStyle.DEFAULT):
        """Initialize the conversation style plugin."""
        self.kernel = kernel
        self.current_style = style
        self._load_style_instructions()

    def _load_style_instructions(self):
        """Load style instruction files."""
        self.style_instructions = {}
        resources_dir = Path(__file__).parent.parent.joinpath("resources")
        
        try:
            # Load casual style
            casual_path = resources_dir / "casual.txt"
            if casual_path.exists():
                with open(casual_path, "r", encoding="utf-8") as f:
                    self.style_instructions[ConversationStyle.CASUAL] = f.read().strip()
            
            # Load GenZ style
            genz_path = resources_dir / "genZ.txt"
            if genz_path.exists():
                with open(genz_path, "r", encoding="utf-8") as f:
                    self.style_instructions[ConversationStyle.GENZ] = f.read().strip()
            
            # Default style has no additional instructions
            self.style_instructions[ConversationStyle.DEFAULT] = ""
            
        except Exception as e:
            print(f"Error loading style instructions: {e}")
            self.style_instructions = {
                ConversationStyle.DEFAULT: "",
                ConversationStyle.CASUAL: "Keep it chill and real, like talking to a buddy.",
                ConversationStyle.GENZ: "Talk like you're on TikTok—use Gen Z slang, keep it hype. Drop those 'lit', 'fam', and 'no cap' vibes. Stay trendy and fresh."
            }

    def set_style(self, style: ConversationStyle) -> bool:
        """Set the current conversation style."""
        if isinstance(style, ConversationStyle):
            self.current_style = style
            return True
        elif isinstance(style, str):
            try:
                self.current_style = ConversationStyle(style.lower())
                return True
            except ValueError:
                return False
        return False

    @kernel_function(description="Get conversation style instructions", name="get_style_instructions")
    def get_style_instructions(self, style: Optional[str] = None) -> str:
        """Get the conversation style instructions."""
        if style:
            try:
                target_style = ConversationStyle(style.lower())
            except ValueError:
                target_style = self.current_style
        else:
            target_style = self.current_style
        
        return self.style_instructions.get(target_style, "")

    @kernel_function(description="Apply conversation style to brand personality", name="enhance_brand_with_style")
    def enhance_brand_with_style(self, brand_instructions: str, style: Optional[str] = None) -> str:
        """Enhance brand personality with conversation style."""
        if style:
            try:
                target_style = ConversationStyle(style.lower())
            except ValueError:
                target_style = self.current_style
        else:
            target_style = self.current_style
        
        style_instructions = self.style_instructions.get(target_style, "")
        
        if not style_instructions:
            return brand_instructions
        
        if not brand_instructions:
            return f"CONVERSATION STYLE:\n{style_instructions}"
        
        return f"{brand_instructions}\n\nCONVERSATION STYLE:\n{style_instructions}"

    @kernel_function(description="Get current conversation style", name="get_current_style")
    def get_current_style(self) -> str:
        """Get the current conversation style."""
        return self.current_style.value

    def list_available_styles(self) -> Dict[str, str]:
        """List all available conversation styles with descriptions."""
        return {
            ConversationStyle.DEFAULT.value: "Standard brand personality only",
            ConversationStyle.CASUAL.value: "Casual, friendly, buddy-like conversation",
            ConversationStyle.GENZ.value: "Gen Z slang, TikTok vibes, trendy language"
        }