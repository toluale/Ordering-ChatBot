import json
import logging
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union, Type, cast
from pathlib import Path

from pydantic import BaseModel, Discriminator, Tag, model_serializer, model_validator, Field
from typing_extensions import Annotated

from .menu_manager import get_menu_manager, MenuManager

logger = logging.getLogger(__name__)


class MenuContext:
    """Context manager for brand-specific menu operations."""
    
    def __init__(self):
        self.current_brand: Optional[str] = None
        self.menu_manager = get_menu_manager()
        self._product_codes: Dict[str, str] = {}
        self._name_to_product_code: Dict[str, str] = {}
        self._toppings_codes: Dict[str, Dict[str, str]] = {}
        self._name_to_topping_code: Dict[str, str] = {}
        self._amount_codes: Dict[str, str] = {}
        self._default_toppings: Dict[str, List[str]] = {}
        
        # Initialize from environment if available
        self._initialize_from_environment()
    
    def _initialize_from_environment(self):
        """Initialize brand context from environment variables."""
        try:
            # Check if menu manager has a current brand set from environment
            current_brand = self.menu_manager.get_current_brand()
            if current_brand:
                self.set_brand(current_brand)
                logger.info(f"Initialized MenuContext with brand from environment: {current_brand}")
        except Exception as e:
            logger.warning(f"Could not initialize from environment: {e}")
    
    def set_brand(self, brand_name: str):
        """Set the current brand and load its menu configuration."""
        if not brand_name:
            raise ValueError("Brand name cannot be None or empty")
        
        self.current_brand = brand_name
        self._load_brand_data()
    
    def require_brand(self) -> str:
        """Get current brand or raise error if not set."""
        if not self.current_brand:
            raise ValueError(
                "No brand context set "
            )
        return self.current_brand
    
    def _load_brand_data(self):
        """Load menu data for the current brand."""
        if not self.current_brand:
            raise ValueError("No current brand set")
        
        try:
            # Ensure menu manager has this brand set
            self.menu_manager.set_current_brand(self.current_brand)
            
            # Generate product codes for the brand
            self._product_codes = self.menu_manager.generate_product_codes(self.current_brand)
            self._name_to_product_code = {v: k for k, v in self._product_codes.items()}
            
            # Load toppings
            self._toppings_codes = self.menu_manager.get_toppings_for_brand(self.current_brand)
            self._name_to_topping_code = {
                info["name"]: code for code, info in self._toppings_codes.items()
            }
            
            # Load menu config
            menu_config = self.menu_manager.require_current_menu_config()
            
            # Get amount codes with brand-specific values, no defaults
            self._amount_codes = menu_config.get("amount_codes", {})
            if not self._amount_codes:
                logger.warning(f"No amount codes defined for brand {self.current_brand}")
            
            # Load default toppings for each category
            item_types = menu_config.get("item_types", {})
            self._default_toppings = {
                category: info.get("default_toppings", [])
                for category, info in item_types.items()
            }
            
            logger.info(f"Loaded menu data for brand: {self.current_brand}")
            
        except Exception as e:
            logger.error(f"Error loading brand data for {self.current_brand}: {e}")
            raise RuntimeError(f"Failed to load menu data for brand: {self.current_brand}") from e
    
    def get_product_codes(self) -> Dict[str, str]:
        """Get product codes for current brand."""
        return self._product_codes.copy()
    
    def get_name_to_product_code(self) -> Dict[str, str]:
        """Get name to product code mapping for current brand."""
        return self._name_to_product_code.copy()
    
    def get_toppings_codes(self) -> Dict[str, Dict[str, str]]:
        """Get toppings codes for current brand."""
        return self._toppings_codes.copy()
    
    def get_name_to_topping_code(self) -> Dict[str, str]:
        """Get name to topping code mapping for current brand."""
        return self._name_to_topping_code.copy()
    
    def get_amount_codes(self) -> Dict[str, str]:
        """Get amount codes for current brand."""
        return self._amount_codes.copy()
    
    def get_default_toppings(self, category: str) -> List[str]:
        """Get default toppings for an item category."""
        return self._default_toppings.get(category, []).copy()
    
    def is_valid_product(self, description: str) -> bool:
        """Check if a product description is valid for current brand."""
        return description in self._name_to_product_code
    
    def is_valid_topping(self, name: str) -> bool:
        """Check if a topping name is valid for current brand."""
        return name.lower() in self._name_to_topping_code
    
    def get_product_code(self, description: str) -> Optional[str]:
        """Get product code for description."""
        return self._name_to_product_code.get(description)
    
    def get_topping_code(self, name: str) -> Optional[str]:
        """Get topping code for name."""
        return self._name_to_topping_code.get(name.lower())


# Global menu context
_menu_context = MenuContext()

def get_menu_context() -> MenuContext:
    """Get the global menu context."""
    return _menu_context

def set_brand_context(brand_name: str):
    """Set the current brand for menu operations."""
    _menu_context.set_brand(brand_name)

def get_current_brand() -> Optional[str]:
    """Get the currently set brand."""
    return _menu_context.current_brand


class LLMToppingAmount(str, Enum):
    none = "none"
    half = "half"
    light = "light"
    normal = "normal"
    double = "double"
    extra = "extra"


class Topping(BaseModel):
    code: str
    amount: Literal["0", "0.5", "1", "2"]


class Item(BaseModel):
    code: str
    toppings: List[Topping] = []
    quantity: int
    description: Optional[str] = None

    @model_validator(mode="after")
    def create_description(self):
        context = get_menu_context()
        product_codes = context.get_product_codes()
        toppings_codes = context.get_toppings_codes()
        
        # Code to amount mapping
        code_to_amount = {"0": "no ", "0.5": "light ", "1": "", "2": "extra "}
        
        self.toppings = list(sorted(self.toppings, key=lambda x: x.code))
        if len(self.toppings) == 0:
            self.description = f"{self.quantity} - {product_codes.get(self.code, 'Unknown item')}"
        else:
            topping_descriptions = []
            for t in self.toppings:
                topping_info = toppings_codes.get(t.code, {})
                topping_name = topping_info.get("name", "Unknown topping")
                amount_prefix = code_to_amount.get(t.amount, "")
                topping_descriptions.append(f"{amount_prefix}{topping_name}")
            
            self.description = f"{self.quantity} - {product_codes.get(self.code, 'Unknown item')} with {', '.join(topping_descriptions)}"
        return self


class Order(BaseModel):
    items: List[Item]


class LLMTopping(BaseModel):
    name: str = Field(description="Name of topping")
    amount: str = Field(description="Amount of topping")

    def __init__(self, **kwargs):
        # Set default amount from current brand context
        if 'amount' not in kwargs:
            context = get_menu_context()
            amount_codes = context.get_amount_codes()
            # Find "normal" amount or use first available amount
            if "normal" in amount_codes:
                kwargs['amount'] = "normal"
            elif amount_codes:
                kwargs['amount'] = next(iter(amount_codes.keys()))
            else:
                raise ValueError(f"No amount codes defined for brand {context.current_brand}")
        super().__init__(**kwargs)

    def __str__(self):
        return f"{self.name.lower()}, {self.amount.lower()}"

    def to_order_item(self):
        """Convert to Topping using current brand context."""
        context = get_menu_context()
        try:
            topping_code = context.get_topping_code(self.name.lower())
            if not topping_code:
                logger.warning(f"Invalid topping for {context.current_brand}: {self.name}")
                return None
            
            amount_codes = context.get_amount_codes()
            amount_code = amount_codes.get(self.amount)
            if not amount_code:
                logger.warning(f"Invalid amount for {context.current_brand}: {self.amount}")
                return None
            
            # Ensure amount_code is valid Literal type and cast it properly
            valid_amounts = {"0", "0.5", "1", "2"}
            if amount_code in valid_amounts:
                final_amount = cast(Literal["0", "0.5", "1", "2"], amount_code)
            else:
                logger.warning(f"Invalid amount code: {amount_code}")
                return None
            
            return Topping(code=topping_code, amount=final_amount)
        except KeyError:
            logger.warning(f"Invalid topping: {self.name}")
            return None


class LLMBurgerItem(BaseModel):
    name: str = Field(description="Name of burger")
    toppings: Optional[List[LLMTopping]] = Field(default=None, description="Toppings of item")
    size: Optional[str] = Field(default=None, description="Size of item")
    bun: Optional[str] = Field(default=None, description="Bun of item")
    patties: Optional[str] = Field(default=None, description="Number of patties")
    cook: Optional[str] = Field(default=None, description="Cook level of item")
    quantity: int = Field(default=1, description="Quantity of item")

    def __init__(self, **kwargs):
        # Set defaults for burger-specific fields
        defaults = {
            'size': DefaultValueHelper.get_default_size("burger"),
            'bun': DefaultValueHelper.get_default_bun(),
            'patties': DefaultValueHelper.get_default_patties(),
            'cook': DefaultValueHelper.get_default_cook()
        }
        
        for field, default_value in defaults.items():
            if field not in kwargs or kwargs[field] is None:
                kwargs[field] = default_value
        
        super().__init__(**kwargs)

    def __str__(self):
        """Convert to description format compatible with menu system."""
        context = get_menu_context()
        if not context.current_brand:
            # Fallback to basic format if no brand context
            parts = [self.name.lower()]
            for field in ['size', 'patties', 'bun', 'cook']:
                value = getattr(self, field, None)
                if value:
                    parts.append(value.lower())
            return ", ".join(parts)
        
        # Get menu configuration for proper mapping
        try:
            menu_config = context.menu_manager.require_current_menu_config()
            item_types = menu_config.get("item_types", {})
            burger_config = item_types.get("burger", {})
            
            # Get mapping configurations
            size_mapping = burger_config.get("size_mapping", {})
            patties_mapping = burger_config.get("patties_mapping", {})
            buns_mapping = burger_config.get("buns_mapping", {})
            cook_mapping = burger_config.get("cook_mapping", {})
            
            # Create reverse mappings for descriptive to code conversion
            reverse_size = {v.lower(): k for k, v in size_mapping.items()}
            reverse_patties = {v.lower(): k for k, v in patties_mapping.items()}
            reverse_buns = {v.lower(): k for k, v in buns_mapping.items()}
            reverse_cook = {v.lower(): k for k, v in cook_mapping.items()}
            
            # Convert descriptive values to codes, then back to proper descriptions
            parts = [self.name.lower()]
            
            # Handle size conversion
            size_val = getattr(self, 'size', None)
            if size_val:
                size_lower = size_val.lower()
                # Try direct mapping first, then reverse mapping
                if size_lower in reverse_size:
                    code = reverse_size[size_lower]
                    parts.append(size_mapping.get(code, size_val).lower())
                elif size_val in size_mapping:
                    parts.append(size_mapping[size_val].lower())
                else:
                    # Handle common LLM terms that don't match menu terms
                    if size_lower in ['default', 'regular', 'standard', 'medium']:
                        # Use quarter lb (14) as default since it's most common
                        parts.append(size_mapping.get('14', 'quarter lb').lower())
                    elif size_lower in ['large', 'big']:
                        # Use half lb (13) for large
                        parts.append(size_mapping.get('13', 'half lb').lower())
                    elif size_lower in ['small']:
                        # Use quarter lb (14) for small  
                        parts.append(size_mapping.get('14', 'quarter lb').lower())
                    else:
                        # If size_mapping has values, use first as default
                        if size_mapping:
                            first_code = next(iter(size_mapping.keys()))
                            parts.append(size_mapping[first_code].lower())
                        else:
                            parts.append(size_val.lower())
            
            # Handle patties conversion  
            patties_val = getattr(self, 'patties', None)
            if patties_val:
                patties_lower = patties_val.lower()
                if patties_lower in reverse_patties:
                    code = reverse_patties[patties_lower]
                    parts.append(patties_mapping.get(code, patties_val).lower())
                elif patties_val in patties_mapping:
                    parts.append(patties_mapping[patties_val].lower())
                else:
                    parts.append(patties_val.lower())
            
            # Handle bun conversion
            bun_val = getattr(self, 'bun', None)  
            if bun_val:
                bun_lower = bun_val.lower()
                if bun_lower in reverse_buns:
                    code = reverse_buns[bun_lower]
                    parts.append(buns_mapping.get(code, bun_val).lower())
                elif bun_val in buns_mapping:
                    parts.append(buns_mapping[bun_val].lower())
                else:
                    # Handle common LLM terms for buns
                    if bun_lower in ['default', 'regular', 'standard', 'normal']:
                        # Use sesame (S) as default  
                        parts.append(buns_mapping.get('S', 'sesame').lower())
                    elif bun_lower in ['pretzel', 'pretzel bun']:
                        # Use pretzel (P)
                        parts.append(buns_mapping.get('P', 'pretzel').lower())
                    elif bun_lower in ['sesame', 'sesame bun']:
                        # Use sesame (S)
                        parts.append(buns_mapping.get('S', 'sesame').lower())
                    else:
                        # Default to sesame if mapping exists
                        if buns_mapping:
                            parts.append(buns_mapping.get('S', 'sesame').lower())
                        else:
                            parts.append(bun_val.lower())
            
            # Handle cook conversion
            cook_val = getattr(self, 'cook', None)
            if cook_val:
                cook_lower = cook_val.lower().replace(' ', '-').replace('_', '-')
                if cook_lower in reverse_cook:
                    code = reverse_cook[cook_lower]
                    parts.append(cook_mapping.get(code, cook_val).lower())
                elif cook_val in cook_mapping:
                    parts.append(cook_mapping[cook_val].lower()) 
                else:
                    parts.append(cook_val.lower())
            
            return ", ".join(parts)
            
        except Exception as e:
            logger.warning(f"Error converting item to description format: {e}")
            # Fallback to basic format
            parts = [self.name.lower()]
            for field in ['size', 'patties', 'bun', 'cook']:
                value = getattr(self, field, None)
                if value:
                    parts.append(value.lower())
            return ", ".join(parts)

    @model_validator(mode="after")
    def validate_toppings(self):
        """Validate toppings using helper."""
        self.toppings = ToppingValidationHelper.validate_toppings(self.toppings, "burger")
        return self

    @model_serializer
    def ser_model(self):
        """Serialize using helper with configuration-driven field handling."""
        # Build extra fields dict for burger-specific fields
        extra_fields = {}
        
        # Handle burger-specific fields with their defaults
        burger_fields = ['bun', 'patties', 'cook']
        for field in burger_fields:
            value = getattr(self, field, None)
            if value:
                default_value = DefaultValueHelper.get_default_for_field("burger", field)
                if value != default_value:
                    extra_fields[field] = value
        
        # Handle toppings with brand-specific logic
        if self.toppings:
            context = get_menu_context()
            default_toppings_codes = context.get_default_toppings("burger")
            toppings_codes = context.get_toppings_codes()
            
            default_toppings_names = [
                toppings_codes.get(code, {}).get("name", "").lower()
                for code in default_toppings_codes
            ]
            
            default_amount = ToppingValidationHelper.get_default_amount()
            
            toppings = []
            for topping in self.toppings:
                if (topping.amount != default_amount if default_amount else True) or topping.name.lower() not in default_toppings_names:
                    toppings.append(topping.model_dump())
            if toppings:
                extra_fields["toppings"] = toppings
        
        return SerializationHelper.serialize_with_field_defaults(self, "burger", ["size"], extra_fields)

    def to_order_item(self):
        """Convert using helper."""
        toppings = []
        if self.toppings:
            for topping in self.toppings:
                try:
                    order_topping = topping.to_order_item()
                    if order_topping:
                        toppings.append(order_topping)
                except (KeyError, ValueError):
                    continue
        
        return ItemConversionHelper.convert_to_order_item(self, toppings)


class LLMGenericItem(BaseModel):
    """Generic item for non-burger menu items (pizza, tacos, etc.)"""
    name: str = Field(description="Name of item")
    category: str = Field(description="Category of item (pizza, taco, etc.)")
    size: Optional[str] = Field(default=None, description="Size of item")
    protein: Optional[str] = Field(default=None, description="Protein choice")
    flavor: Optional[str] = Field(default=None, description="Flavor choice")
    toppings: Optional[List[LLMTopping]] = Field(default=None, description="Toppings/modifications")
    quantity: int = Field(default=1, description="Quantity of item")

    def __init__(self, **kwargs):
        category = kwargs.get('category', 'generic')
        
        # Set defaults for configurable fields
        defaults = {
            'size': DefaultValueHelper.get_default_size(category),
            'protein': DefaultValueHelper.get_default_for_field(category, 'protein'),
            'flavor': DefaultValueHelper.get_default_for_field(category, 'flavor')
        }
        
        for field, default_value in defaults.items():
            if field not in kwargs or kwargs[field] is None:
                kwargs[field] = default_value
        
        super().__init__(**kwargs)

    def __str__(self):
        parts = [self.name.lower()]
        for field in ['size', 'protein', 'flavor']:
            value = getattr(self, field, None)
            if value:
                parts.append(value.lower())
        return ", ".join(parts)

    @model_validator(mode="after")
    def validate_toppings(self):
        """Validate toppings using helper."""
        self.toppings = ToppingValidationHelper.validate_toppings(self.toppings, self.category)
        return self

    @model_serializer
    def ser_model(self):
        """Serialize using helper with configuration-driven field handling."""
        extra_fields = {}
        
        # Handle category-specific fields with their defaults
        for field in ['protein', 'flavor']:
            value = getattr(self, field, None)
            if value:
                default_value = DefaultValueHelper.get_default_for_field(self.category, field)
                if value != default_value:
                    extra_fields[field] = value
        
        # Handle toppings
        if self.toppings and len(self.toppings) > 0:
            extra_fields["toppings"] = [topping.model_dump() for topping in self.toppings]
        
        return SerializationHelper.serialize_with_field_defaults(self, self.category, ["size"], extra_fields)

    def to_order_item(self):
        """Convert using helper."""
        toppings = []
        if self.toppings:
            for topping in self.toppings:
                try:
                    order_topping = topping.to_order_item()
                    if order_topping:
                        toppings.append(order_topping)
                except (KeyError, ValueError):
                    continue
        
        return ItemConversionHelper.convert_to_order_item(self, toppings)


class LLMFriesItem(BaseModel):
    name: str = Field(description="Name of fries")
    size: Optional[str] = Field(default=None, description="Size of item")
    toppings: Optional[List[LLMTopping]] = Field(default=None, description="Salt option")
    quantity: int = Field(default=1, description="Quantity of item")

    def __init__(self, **kwargs):
        # Set size default using helper
        if 'size' not in kwargs or kwargs['size'] is None:
            kwargs['size'] = DefaultValueHelper.get_default_size("side")
        
        super().__init__(**kwargs)

    @model_validator(mode="after")
    def validate_toppings(self):
        """Validate salt toppings using helper."""
        self.toppings = ToppingValidationHelper.validate_fries_salt_toppings(self.toppings)
        return self

    def __str__(self):
        """Convert to description format compatible with menu system."""
        context = get_menu_context()
        if not context.current_brand:
            # Fallback to basic format - fries only have salt variants, no size
            salt_str = ItemConversionHelper.get_fries_salt_description(self.toppings)
            return f"{self.name.lower()}, {salt_str}"
        
        try:
            # For fries, the menu system only uses salt variants, not size
            # Format: "fries, salted" or "fries, unsalted"
            salt_str = ItemConversionHelper.get_fries_salt_description(self.toppings)
            return f"{self.name.lower()}, {salt_str}"
            
        except Exception as e:
            logger.warning(f"Error converting fries item to description format: {e}")
            # Fallback to basic format
            salt_str = ItemConversionHelper.get_fries_salt_description(self.toppings)
            return f"{self.name.lower()}, {salt_str}"

    @model_serializer
    def ser_model(self):
        """Serialize using helper."""
        return SerializationHelper.serialize_fries_item(self)

    def to_order_item(self):
        """Convert using helper."""
        return ItemConversionHelper.convert_to_order_item(self)


class LLMDrinkItem(BaseModel):
    name: str = Field(description="Name of drink")
    size: Optional[str] = Field(default=None, description="Size of item")
    quantity: int = Field(default=1, description="Quantity of item")

    def __init__(self, **kwargs):
        # Set size default using helper
        if 'size' not in kwargs or kwargs['size'] is None:
            kwargs['size'] = DefaultValueHelper.get_default_size("drink")
        
        super().__init__(**kwargs)

    def __str__(self):
        """Convert to description format compatible with menu system."""
        # For drinks, the menu system only uses flavor names, not size
        # Format: "cola", "diet cola", "lemon-lime", "root beer"
        return self.name.lower()

    @model_serializer
    def ser_model(self):
        """Serialize using helper."""
        return SerializationHelper.serialize_with_field_defaults(self, "drink", ["size"])

    def to_order_item(self):
        """Convert using helper."""
        return ItemConversionHelper.convert_to_order_item(self)


def item_discriminator(item: Any) -> str:
    """Returns item type based on name and current brand context."""
    name_obj = item.get("name") if isinstance(item, dict) else item.name
    name = name_obj.lower() if name_obj else ""
    
    # Get current brand's menu to determine item categories
    context = get_menu_context()
    if not context.current_brand:
        raise ValueError("No brand context set. Cannot determine item type without brand configuration.")
    
    menu_config = context.menu_manager.load_brand_menu(context.current_brand)
    item_categories = menu_config.get("brand_info", {}).get("item_categories", [])
    
    # Check against brand's menu items
    menu_items = menu_config.get("menu_items", {})
    for item_name, item_config in menu_items.items():
        name_variations = item_config.get("name_variations", [item_name])
        if any(variation.lower() in name for variation in name_variations):
            category = item_config.get("category", "generic")
            if category == "burger":
                return "burger"
            elif category in ["side"] and "fries" in name:
                return "fries"
            elif category == "drink":
                return "drink"
            else:
                return "generic"
    
    # If no specific item match found, return generic
    # This ensures all items are handled through brand configuration
    logger.warning(f"Item '{name}' not found in {context.current_brand} menu configuration. Using generic type.")
    return "generic"


LLMItem = Annotated[
    Union[
        Annotated[LLMFriesItem, Tag("fries")],
        Annotated[LLMDrinkItem, Tag("drink")],
        Annotated[LLMBurgerItem, Tag("burger")],
        Annotated[LLMGenericItem, Tag("generic")],
    ],
    Discriminator(item_discriminator),
]


class LLMOrder(BaseModel):
    items: List[LLMItem]

    @model_validator(mode="after")
    def validate_against_order(self):
        """Validate items against order schema and keep only valid items."""
        valid_items = []
        for item in self.items:
            try:
                valid_item = item.to_order_item()
                if valid_item:
                    valid_items.append(item)
            except Exception:
                continue
        self.items = valid_items
        return self

    @model_serializer
    def ser_model(self):
        """Serialize LLMOrder to dictionary."""
        return {"items": [item.model_dump() for item in self.items]}

    def to_order(self):
        """Convert LLMOrder to Order."""
        valid_order_items = []
        for item in self.items:
            order_item = item.to_order_item()
            if order_item:
                valid_order_items.append(order_item)
        return Order(items=valid_order_items)


# Convenience functions for backward compatibility and easy access
def get_product_codes() -> Dict[str, str]:
    """Get product codes for current brand."""
    return get_menu_context().get_product_codes()

def get_name_to_product_code() -> Dict[str, str]:
    """Get name to product code mapping for current brand."""
    return get_menu_context().get_name_to_product_code()

def get_toppings_codes() -> Dict[str, Dict[str, str]]:
    """Get toppings codes for current brand."""
    return get_menu_context().get_toppings_codes()

def get_name_to_topping_code() -> Dict[str, str]:
    """Get name to topping code mapping for current brand."""
    return get_menu_context().get_name_to_topping_code()

def get_amount_codes() -> Dict[str, str]:
    """Get amount codes for current brand."""
    return get_menu_context().get_amount_codes()

def get_default_toppings(category: str) -> List[str]:
    """Get default toppings for an item category."""
    return get_menu_context().get_default_toppings(category)


class MenuConfigHelper:
    """Helper class for accessing menu configuration data generically."""
    
    @staticmethod
    def get_menu_config() -> Dict[str, Any]:
        """Get the current menu configuration."""
        context = get_menu_context()
        return context.menu_manager.require_current_menu_config()
    
    @staticmethod
    def get_category_config(category: str) -> Dict[str, Any]:
        """Get configuration for a specific item category."""
        menu_config = MenuConfigHelper.get_menu_config()
        return menu_config.get("item_types", {}).get(category, {})
    
    @staticmethod
    def get_field_mapping(category: str, mapping_key: str) -> Dict[str, str]:
        """Get field mapping for a category (e.g., size_mapping, buns_mapping)."""
        category_config = MenuConfigHelper.get_category_config(category)
        return category_config.get(mapping_key, {})
    
    @staticmethod
    def get_default_value_code(category: str, field: str) -> Optional[str]:
        """Get the default code for a specific field in a category."""
        menu_config = MenuConfigHelper.get_menu_config()
        menu_items = menu_config.get("menu_items", {})
        
        for item_config in menu_items.values():
            if item_config.get("category") == category:
                return item_config.get("default", {}).get(field)
        return None


class DefaultValueHelper:
    """Helper class for setting default values based on brand configuration."""
    
    @staticmethod
    def get_default_value(category: str, field: str, mapping_key: str) -> Optional[str]:
        """Generic method to get default values from menu configuration."""
        # Get the mapping for the specific field
        field_mapping = MenuConfigHelper.get_field_mapping(category, mapping_key)
        if not field_mapping:
            return None
            
        # Look for default value in menu items
        default_code = MenuConfigHelper.get_default_value_code(category, field)
        if default_code and default_code in field_mapping:
            return field_mapping[default_code]
        
        # Return first available value as fallback
        return next(iter(field_mapping.values())) if field_mapping else None
    
    @staticmethod
    def get_default_size(category: str) -> Optional[str]:
        """Get default size for a given item category."""
        return DefaultValueHelper.get_default_value(category, "size", "size_mapping")
    
    @staticmethod
    def get_default_bun() -> Optional[str]:
        """Get default bun for burger items."""
        return DefaultValueHelper.get_default_value("burger", "buns", "buns_mapping")
    
    @staticmethod
    def get_default_patties() -> Optional[str]:
        """Get default patties for burger items."""
        return DefaultValueHelper.get_default_value("burger", "patties", "patties_mapping")
    
    @staticmethod
    def get_default_cook() -> Optional[str]:
        """Get default cook level for burger items."""
        return DefaultValueHelper.get_default_value("burger", "cook", "cook_mapping")
    
    @staticmethod
    def get_default_for_field(category: str, field: str) -> Optional[str]:
        """Get default value for any field by inferring the mapping key."""
        # Common mapping patterns
        mapping_patterns = {
            "size": "size_mapping",
            "buns": "buns_mapping", 
            "bun": "buns_mapping",
            "patties": "patties_mapping",
            "cook": "cook_mapping",
            "protein": "protein_mapping",
            "flavor": "flavor_mapping",
            "salt": "salt_mapping"
        }
        
        mapping_key = mapping_patterns.get(field, f"{field}_mapping")
        return DefaultValueHelper.get_default_value(category, field, mapping_key)


class SerializationHelper:
    """Helper class for common serialization logic."""
    
    @staticmethod
    def serialize_with_defaults(item, category: str, extra_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Serialize item excluding default values."""
        model = {"name": item.name, "quantity": item.quantity}
        
        # Add size if not default
        default_size = DefaultValueHelper.get_default_size(category)
        if hasattr(item, 'size') and item.size and item.size != default_size:
            model["size"] = item.size
        
        # Add extra fields
        if extra_fields:
            for field_name, field_value in extra_fields.items():
                if field_value is not None:
                    model[field_name] = field_value
        
        return model
    
    @staticmethod
    def serialize_with_field_defaults(item, category: str, fields: List[str], extra_fields: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Serialize item with specific fields checked against defaults."""
        model = {"name": item.name, "quantity": item.quantity}
        
        # Check each field against its default
        for field in fields:
            if hasattr(item, field):
                field_value = getattr(item, field)
                if field_value is not None:
                    default_value = DefaultValueHelper.get_default_for_field(category, field)
                    if field_value != default_value:
                        model[field] = field_value
        
        # Add extra fields
        if extra_fields:
            for field_name, field_value in extra_fields.items():
                if field_value is not None:
                    model[field_name] = field_value
        
        return model
    
    @staticmethod
    def serialize_fries_item(item) -> Dict[str, Any]:
        """Serialize fries item with salt handling."""
        model = {"name": item.name, "quantity": item.quantity}
        
        # Add size if not default
        default_size = DefaultValueHelper.get_default_size("side")
        if item.size and item.size != default_size:
            model["size"] = item.size
        
        # Handle salt toppings
        if item.toppings and len(item.toppings) > 0:
            default_amount = ToppingValidationHelper.get_default_amount()
            if item.toppings[0].amount != default_amount:
                model["amount"] = item.toppings[0].amount
        
        return model


class ToppingValidationHelper:
    """Helper class for topping validation logic."""
    
    @staticmethod
    def get_default_amount() -> Optional[str]:
        """Get the default amount name (usually 'normal')."""
        context = get_menu_context()
        amount_codes = context.get_amount_codes()
        for amount_name, amount_code in amount_codes.items():
            if amount_code == "1":  # normal amount
                return amount_name
        return None
    
    @staticmethod
    def get_default_toppings_for_category(category: str) -> List[str]:
        """Get default topping names for a category."""
        context = get_menu_context()
        default_toppings_codes = context.get_default_toppings(category)
        toppings_codes = context.get_toppings_codes()
        
        # Convert topping codes to names for default toppings
        default_toppings_names = []
        for code in default_toppings_codes:
            topping_info = toppings_codes.get(code, {})
            if "name" in topping_info:
                default_toppings_names.append(topping_info["name"])
        
        return default_toppings_names
    
    @staticmethod
    def validate_toppings(toppings: Optional[List[LLMTopping]], category: str) -> List[LLMTopping]:
        """Validate and filter toppings for a given category."""
        default_toppings_names = ToppingValidationHelper.get_default_toppings_for_category(category)
        
        if toppings is None:
            toppings = [LLMTopping(name=name) for name in default_toppings_names]
        else:
            # Fill in missing default toppings
            current_toppings = [topping.name.lower() for topping in toppings]
            missing_toppings = [
                LLMTopping(name=name)
                for name in default_toppings_names
                if name.lower() not in current_toppings
            ]
            toppings += missing_toppings

        # Only keep validated toppings
        valid_toppings = []
        for item in toppings:
            try:
                valid_item = item.to_order_item()
                if valid_item:
                    valid_toppings.append(item)
            except (ValueError, KeyError):
                logger.warning("Invalid topping: %s", str(item))
                continue
        
        return valid_toppings
    
    @staticmethod
    def get_default_salt_setting(category: str = "side") -> Optional[str]:
        """Get default salt setting from menu configuration."""
        menu_config = MenuConfigHelper.get_menu_config()
        menu_items = menu_config.get("menu_items", {})
        
        for item_config in menu_items.values():
            if item_config.get("category") == category and "fries" in item_config.get("name_variations", []):
                default_salt_code = item_config.get("default", {}).get("salt")
                if default_salt_code:
                    salt_mapping = MenuConfigHelper.get_field_mapping(category, "salt_mapping")
                    return salt_mapping.get(default_salt_code)
                break
        return None
    
    @staticmethod
    def validate_fries_salt_toppings(toppings: Optional[List[LLMTopping]]) -> List[LLMTopping]:
        """Special validation for fries salt toppings."""
        context = get_menu_context()
        toppings_codes = context.get_toppings_codes()
        salt_available = any("salt" in info.get("name", "").lower() for info in toppings_codes.values())
        
        if not salt_available:
            return []
        
        # Get default amount for salt
        default_amount = ToppingValidationHelper.get_default_amount()
        if not default_amount:
            return []
        
        if toppings is None:
            # Get default salt setting from menu config
            default_salt_name = ToppingValidationHelper.get_default_salt_setting()
            if default_salt_name:
                return [LLMTopping(name="salt", amount=default_amount)]
            return []
        
        # Only allow salt-related toppings
        salt_toppings = [
            topping for topping in toppings 
            if "salt" in topping.name.lower()
        ]
        
        if salt_toppings:
            return [salt_toppings[0]]
        else:
            # Set default salt topping
            return [LLMTopping(name="salt", amount=default_amount)]


class ItemConversionHelper:
    """Helper class for converting items to order format."""
    
    @staticmethod
    def convert_to_order_item(item, toppings: Optional[List[Topping]] = None) -> Optional[Item]:
        """Convert an LLM item to Order item format."""
        context = get_menu_context()
        try:
            product_code = context.get_product_code(str(item))
            if not product_code:
                logger.warning(f"Invalid item for {context.current_brand}: {str(item)}")
                return None
            
            return Item(
                code=product_code,
                toppings=toppings or [],
                quantity=item.quantity,
            )
        except KeyError:
            logger.warning(f"Invalid item: {str(item)}")
            return None
    
    @staticmethod
    def get_category_salt_description(toppings: Optional[List[LLMTopping]], category: str = "side") -> str:
        """Generate salt description for a category (generalized from fries)."""
        context = get_menu_context()
        toppings_codes = context.get_toppings_codes()
        salt_available = any("salt" in info.get("name", "").lower() for info in toppings_codes.values())
        
        if salt_available and toppings and isinstance(toppings, list) and len(toppings) > 0:
            salt_amount = toppings[0]
            default_amount = ToppingValidationHelper.get_default_amount()
            return "salted" if salt_amount.amount == default_amount else "unsalted"
        else:
            # Default salt status from menu config if available
            default_salt = ToppingValidationHelper.get_default_salt_setting(category)
            return default_salt if default_salt else "salted"
    
    @staticmethod
    def get_fries_salt_description(toppings: Optional[List[LLMTopping]]) -> str:
        """Generate salt description for fries (backward compatibility)."""
        return ItemConversionHelper.get_category_salt_description(toppings, "side")
