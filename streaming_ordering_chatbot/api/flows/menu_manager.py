import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from enum import Enum
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class MenuItemType(str, Enum):
    """Standard menu item types across all brands."""
    BURGER = "burger"
    PIZZA = "pizza" 
    SANDWICH = "sandwich"
    WRAP = "wrap"
    BOWL = "bowl"
    TACO = "taco"
    BURRITO = "burrito"
    QUESADILLA = "quesadilla"
    SIDE = "side"
    FRIES = "fries"
    NACHOS = "nachos"
    SALAD = "salad"
    DRINK = "drink"
    DESSERT = "dessert"
    APPETIZER = "appetizer"


@dataclass
class MenuItemConfig:
    """Configuration for a menu item."""
    code: str
    name: str
    category: MenuItemType
    base_description: str
    allowed_sizes: List[str]
    allowed_toppings: List[str]
    default_size: str
    default_toppings: List[str]
    customizable: bool = True


@dataclass
class ToppingConfig:
    """Configuration for a topping/modification."""
    code: str
    name: str
    description: str
    category: str 
    applicable_items: List[MenuItemType]


class MenuManager:
    """Manages menus for different restaurant brands."""
    
    def __init__(self, data_dir: Optional[str] = None):
        """Initialize menu manager.
        
        Args:
            data_dir: Directory containing menu data files
        """
        self.data_dir = Path(data_dir) if data_dir else Path(__file__).parent / "data"
        self.brand_menus: Dict[str, Dict[str, Any]] = {}
        self.loaded_configs: Dict[str, Dict[str, Any]] = {}
        
        # Current brand configuration from environment
        self.current_brand: Optional[str] = None
        self.current_menu_config: Optional[Dict[str, Any]] = None
        
        self._load_from_environment()
        
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def _normalize_brand_name(self, brand_name: str) -> str:
        """Normalize brand name for file operations."""
        return brand_name.lower().replace(" ", "_").replace("'", "").replace(".", "")
    
    def _get_brand_file_path(self, brand_name: str) -> Path:
        """Get the file path for a brand's menu configuration."""
        normalized_name = self._normalize_brand_name(brand_name)
        return self.data_dir / f"{normalized_name}_menu.json"
    
    def _get_config_options(self, item_config: Dict[str, Any], option_keys: List[str]) -> Dict[str, List[str]]:
        """Extract configuration options with fallback to empty string list."""
        options = {}
        for key in option_keys:
            options[key] = item_config.get(key, [""])
        return options
    
    def _get_menu_config_for_brand(self, brand_name: str) -> Dict[str, Any]:
        """Get menu configuration for a brand, with caching."""
        return self.load_brand_menu(brand_name)
    
    def _get_mapping_configs(self, menu_config: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
        """Extract all mapping configurations from menu config."""
        item_types = self._get_item_types_config(menu_config)
        burger_config = item_types.get("burger", {})
        side_config = item_types.get("side", {})
        
        return {
            "size_mapping": burger_config.get("size_mapping", {}) or side_config.get("size_mapping", {}),
            "patties_mapping": burger_config.get("patties_mapping", {}),
            "buns_mapping": burger_config.get("buns_mapping", {}),
            "cook_mapping": burger_config.get("cook_mapping", {}),
            "salt_mapping": side_config.get("salt_mapping", {})
        }
    
    def _get_item_types_config(self, menu_config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract item_types configuration from menu config."""
        return menu_config.get("item_types", {})
    
    def _load_from_environment(self):
        """Load brand and menu configuration from environment variables."""
        brand_name = os.getenv("RESTAURANT_BRAND")
        menu_config_path = os.getenv("MENU_CONFIG_PATH")
        
        if brand_name:
            self.current_brand = brand_name
            logger.info(f"Loaded brand from environment: {brand_name}")
            
            if menu_config_path:
                try:
                    config_path = Path(menu_config_path)
                    if config_path.exists():
                        with open(config_path, 'r', encoding='utf-8') as f:
                            self.current_menu_config = json.load(f)
                        if self.current_menu_config:
                            self.loaded_configs[brand_name] = self.current_menu_config
                        logger.info(f"Loaded menu config from {menu_config_path} for {brand_name}")
                    else:
                        logger.warning(f"Menu config path does not exist: {menu_config_path}")
                except Exception as e:
                    logger.error(f"Error loading menu config from {menu_config_path}: {e}")
            
            if not self.current_menu_config:
                try:
                    self.current_menu_config = self.load_brand_menu(brand_name)
                    logger.info(f"Loaded menu config from standard location for {brand_name}")
                except Exception as e:
                    logger.warning(f"Could not load menu config for {brand_name}: {e}")
    
    def set_current_brand(self, brand_name: str, force_reload: bool = False):
        """Set the current active brand."""
        if self.current_brand != brand_name or force_reload:
            self.current_brand = brand_name
            try:
                self.current_menu_config = self.load_brand_menu(brand_name)
                logger.info(f"Set current brand to: {brand_name}")
            except Exception as e:
                logger.error(f"Failed to set current brand to {brand_name}: {e}")
                self.current_menu_config = None
                raise
    
    def get_current_brand(self) -> Optional[str]:
        """Get the current active brand."""
        return self.current_brand
    
    def get_current_menu_config(self) -> Optional[Dict[str, Any]]:
        """Get the current menu configuration."""
        return self.current_menu_config
    
    def require_current_brand(self) -> str:
        """Get current brand or raise error if not set."""
        if not self.current_brand:
            raise ValueError(
                "No current brand set. Please set RESTAURANT_BRAND environment variable "
            )
        return self.current_brand
    
    def require_current_menu_config(self) -> Dict[str, Any]:
        """Get current menu config or raise error if not available."""
        if not self.current_menu_config:
            brand = self.require_current_brand()
            raise ValueError(
                f"No menu configuration available for brand: {brand}. "
                "Please ensure MENU_CONFIG_PATH environment variable is set, "
                "or that a menu file exists in the data directory."
            )
        return self.current_menu_config
        
    def get_available_brands(self) -> List[str]:
        """Get list of available brand menus."""
        brands = []
        for file_path in self.data_dir.glob("*_menu.json"):
            brand_name = file_path.stem.replace("_menu", "").replace("_", " ").title()
            brands.append(brand_name)
        return brands
    
    def load_brand_menu(self, brand_name: str) -> Dict[str, Any]:
        """Load menu configuration for a specific brand.
        """
        if brand_name in self.loaded_configs:
            return self.loaded_configs[brand_name]
        
        file_path = self._get_brand_file_path(brand_name)
        
        if not file_path.exists():
            logger.error(f"Menu file not found for brand: {brand_name}")
            raise FileNotFoundError(f"Menu configuration file not found: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                menu_config = json.load(f)
            
            # Validate menu structure
            if not self._validate_menu_config(menu_config):
                logger.error(f"Invalid menu structure for {brand_name}")
                raise ValueError(f"Invalid menu configuration structure for brand: {brand_name}")
            
            self.loaded_configs[brand_name] = menu_config
            logger.info(f"Loaded menu for brand: {brand_name}")
            return menu_config
            
        except Exception as e:
            logger.error(f"Error loading menu for {brand_name}: {e}")
            raise RuntimeError(f"Failed to load menu configuration for brand: {brand_name}") from e
    
    def _validate_menu_config(self, config: Dict[str, Any]) -> bool:
        """Validate menu configuration structure. that is the loaded JSON file has the required structure."""
        required_sections = ["brand_info", "menu_items", "toppings", "item_types"]
        
        if not all(section in config for section in required_sections):
            return False
        
        # Validate brand_info
        brand_info = config["brand_info"]
        if not all(key in brand_info for key in ["name", "cuisine_type", "item_categories"]):
            return False
        
        # Validate menu_items structure
        menu_items = config["menu_items"]
        if not isinstance(menu_items, dict):
            return False
        
        # Validate toppings structure
        toppings = config["toppings"]
        if not isinstance(toppings, dict):
            return False
        
        return True
    
    def generate_product_codes(self, brand_name: str) -> Dict[str, str]:
        """Generate product codes for a brand's menu items.
        Args: brand_name: Name of the brand
        Returns: Dictionary mapping product codes to descriptions
        """
        menu_config = self._get_menu_config_for_brand(brand_name)
        product_codes = {}
        
        for item_name, item_config in menu_config["menu_items"].items():
            codes = self._generate_item_codes(item_name, item_config)
            product_codes.update(codes)
        
        return product_codes
    
    def _generate_item_codes(self, item_name: str, item_config: Dict[str, Any]) -> Dict[str, str]:
        """Generate all possible product codes for an item."""
        codes = {}
        pattern = item_config["code_pattern"]
        
        # Get all possible combinations using helper
        option_keys = ["sizes", "patties", "buns", "cooks", "salt"]
        options = self._get_config_options(item_config, option_keys)
        
        # Generate all combinations
        for size in options["sizes"]:
            for patties in options["patties"]:
                for bun in options["buns"]:
                    for cook in options["cooks"]:
                        for salt in options["salt"]:
                            try:
                                code = pattern.format(
                                    size=size,
                                    patties=patties,
                                    bun=bun,
                                    cook=cook,
                                    salt=salt
                                )
                                description = self._generate_description(
                                    item_name, size, patties, bun, cook, salt
                                )
                                codes[code] = description
                            except KeyError:
                                # Skip if pattern doesn't match item configuration
                                continue
        
        return codes
    
    def _generate_description(self, item_name: str, size: str, patties: str, bun: str, cook: str, salt: str) -> str:
        """Converts cryptic codes into customer-friendly descriptions using brand-specific mapping tables."""
        if not self.current_brand:
            return item_name  # Fallback to just item name if no brand context
        
        parts = [item_name]
        
        try:
            menu_config = self.require_current_menu_config()
            mappings = self._get_mapping_configs(menu_config)
            
            # Apply mappings using the extracted configurations
            if size and size in mappings["size_mapping"]:
                parts.append(mappings["size_mapping"][size])
            
            if patties and patties in mappings["patties_mapping"]:
                parts.append(mappings["patties_mapping"][patties])
            
            if bun and bun in mappings["buns_mapping"]:
                parts.append(mappings["buns_mapping"][bun])
            
            if cook and cook in mappings["cook_mapping"]:
                parts.append(mappings["cook_mapping"][cook])
            
            if salt and salt in mappings["salt_mapping"]:
                parts.append(mappings["salt_mapping"][salt])
                
        except Exception as e:
            logger.warning(f"Error generating description for {self.current_brand}: {e}")
        
        return ", ".join(parts)
    
    def get_toppings_for_brand(self, brand_name: str) -> Dict[str, Dict[str, str]]:
        """Get toppings configuration for a brand."""
        menu_config = self._get_menu_config_for_brand(brand_name)
        return menu_config.get("toppings", {})
    
    def get_default_toppings_for_item(self, brand_name: str, item_category: str) -> List[str]:
        """Get default toppings for an item category."""
        menu_config = self._get_menu_config_for_brand(brand_name)
        item_types = self._get_item_types_config(menu_config)
        return item_types.get(item_category, {}).get("default_toppings", [])
    
    def is_item_customizable(self, brand_name: str, item_category: str) -> bool:
        """Check if an item category is customizable."""
        menu_config = self._get_menu_config_for_brand(brand_name)
        item_types = self._get_item_types_config(menu_config)
        return item_types.get(item_category, {}).get("customizable", True)
    
    def create_menu_for_brand(self, brand_name: str, menu_config: Dict[str, Any]) -> bool:
        """Create a new menu configuration file for a brand.
        Args:
            brand_name: Name of the brand
            menu_config: Menu configuration dictionary
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self._validate_menu_config(menu_config):
                logger.error(f"Invalid menu configuration for {brand_name}")
                return False
            
            file_path = self._get_brand_file_path(brand_name)
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(menu_config, f, indent=2, ensure_ascii=False)
            
            self.loaded_configs[brand_name] = menu_config
            logger.info(f"Created menu configuration for {brand_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating menu for {brand_name}: {e}")
            return False
    
    def get_menu_text_format(self, brand_name: str) -> str:
        """Get menu in text format for prompts.
        Args:
            brand_name: Name of the brand   
        Returns:
            Menu formatted as text for AI prompts
        """
        menu_config = self._get_menu_config_for_brand(brand_name)
        product_codes = self.generate_product_codes(brand_name)
        toppings = self.get_toppings_for_brand(brand_name)
        
        lines = [f"# {menu_config['brand_info']['name']} Menu\n"]
        
        # Add cuisine type
        lines.append(f"Cuisine Type: {menu_config['brand_info']['cuisine_type']}\n")
        
        # Add menu items
        lines.append("## Menu Items")
        lines.append("ProductCode: Description")
        for code, description in product_codes.items():
            lines.append(f"{code}: {description}")
        
        # Add toppings
        if toppings:
            lines.append("\n## Available Toppings")
            lines.append("ToppingCode: Description")
            for code, topping_info in toppings.items():
                lines.append(f"{code}: {topping_info['name']}")
        
        return "\n".join(lines)


# Global menu manager instance
_menu_manager: Optional[MenuManager] = None

def get_menu_manager() -> MenuManager:
    """Get the global menu manager instance."""
    global _menu_manager
    if _menu_manager is None:
        _menu_manager = MenuManager()
    return _menu_manager

def initialize_menu_manager(data_dir: Optional[str] = None):
    """Initialize the global menu manager."""
    global _menu_manager
    _menu_manager = MenuManager(data_dir)

def set_menu_manager(manager: MenuManager):
    """Set the global menu manager instance."""
    global _menu_manager
    _menu_manager = manager
