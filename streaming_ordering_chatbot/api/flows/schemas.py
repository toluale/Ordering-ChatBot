import logging
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import (BaseModel, Discriminator, Tag, model_serializer,
                      model_validator)
from typing_extensions import Annotated

PRODUCT_CODES = {
    "HB14SPN": "hamburger, quarter lb, single, pretzel, normal",
    "HB14DPN": "hamburger, quarter lb, double, pretzel, normal",
    "HB13SPN": "hamburger, half lb, single, pretzel, normal",
    "HB13DPN": "hamburger, half lb, double, pretzel, normal",
    "CZHB14SPN": "cheeseburger, quarter lb, single, pretzel, normal",
    "CZHB14DPN": "cheeseburger, quarter lb, double, pretzel, normal",
    "CZHB13SPN": "cheeseburger, half lb, single, pretzel, normal",
    "CZHB13DPN": "cheeseburger, half lb, double, pretzel, normal",
    "HB14SSN": "hamburger, quarter lb, single, sesame, normal",
    "HB14DSN": "hamburger, quarter lb, double, sesame, normal",
    "HB13SSN": "hamburger, half lb, single, sesame, normal",
    "HB13DSN": "hamburger, half lb, double, sesame, normal",
    "CZHB14SSN": "cheeseburger, quarter lb, single, sesame, normal",
    "CZHB14DSN": "cheeseburger, quarter lb, double, sesame, normal",
    "CZHB13SSN": "cheeseburger, half lb, single, sesame, normal",
    "CZHB13DSN": "cheeseburger, half lb, double, sesame, normal",
    "HB14SPW": "hamburger, quarter lb, single, pretzel, well-done",
    "HB14DPW": "hamburger, quarter lb, double, pretzel, well-done",
    "HB13SPW": "hamburger, half lb, single, pretzel, well-done",
    "HB13DPW": "hamburger, half lb, double, pretzel, well-done",
    "CZHB14SPW": "cheeseburger, quarter lb, single, pretzel, well-done",
    "CZHB14DPW": "cheeseburger, quarter lb, double, pretzel, well-done",
    "CZHB13SPW": "cheeseburger, half lb, single, pretzel, well-done",
    "CZHB13DPW": "cheeseburger, half lb, double, pretzel, well-done",
    "HB14SSW": "hamburger, quarter lb, single, sesame, well-done",
    "HB14DSW": "hamburger, quarter lb, double, sesame, well-done",
    "HB13SSW": "hamburger, half lb, single, sesame, well-done",
    "HB13DSW": "hamburger, half lb, double, sesame, well-done",
    "CZHB14SSW": "cheeseburger, quarter lb, single, sesame, well-done",
    "CZHB14DSW": "cheeseburger, quarter lb, double, sesame, well-done",
    "CZHB13SSW": "cheeseburger, half lb, single, sesame, well-done",
    "CZHB13DSW": "cheeseburger, half lb, double, sesame, well-done",
    "BBSS": "black bean burger, quarter lb, single, sesame, normal",
    "BBDS": "black bean burger, quarter lb, double, sesame, normal",
    "DSC": "cola, small",
    "DSD": "diet cola, small",
    "DSLL": "lemon-lime, small",
    "DSR": "root beer, small",
    "DMC": "cola, medium",
    "DMD": "diet cola, medium",
    "DMLL": "lemon-lime, medium",
    "DMR": "root bear, medium",
    "DLC": "cola, large",
    "DLD": "diet cola, large",
    "DLLL": "lemon-lime, large",
    "DLR": "root beer, large",
    "FSS": "fries, small, salted",
    "FMS": "fries, medium, salted",
    "FLS": "fries, large, salted",
    "FSNS": "fries, small, unsalted",
    "FMNS": "fries, medium, unsalted",
    "FLNS": "fries, large, unsalted",
}

NAME_TO_PRODUCT_CODE = {v: k for k, v in PRODUCT_CODES.items()}

TOPPINGS_CODES = {
    "T": "tomato",
    "L": "lettuce",
    "O": "onion",
    "P": "pickle",
    "C": "ketchup",
    "M": "mustard",
    "Y": "mayo",
    "R": "relish",
    "B": "bacon",
}

NAME_TO_TOPPING_CODE = {v: k for k, v in TOPPINGS_CODES.items()}
AMOUNT_TO_CODE = {"none": "0", "half": "0.5", "normal": "1", "double": "2"}
CODE_TO_AMOUNT = {"0": "no ", "0.5": "half ", "1": "", "2": "double "}


class Topping(BaseModel):
    code: str
    amount: Literal["0", "0.5", "1", "2"]


class Item(BaseModel):
    code: str
    toppings: list[Topping] = []
    quantity: int
    description: Optional[str] = None

    @model_validator(mode="after")
    def create_description(self):
        self.toppings = list(sorted(self.toppings, key=lambda x: x.code))
        if len(self.toppings) == 0:
            self.description = f"{self.quantity} - {PRODUCT_CODES[self.code]}"
        else:
            self.description = f"{self.quantity} - {PRODUCT_CODES[self.code]} with {', '.join([CODE_TO_AMOUNT[t.amount] + TOPPINGS_CODES[t.code] for t in self.toppings])}"
        return self


class Order(BaseModel):
    items: list[Union[Item]]


class LLMToppingAmount(str, Enum):
    none = "none"
    half = "half"
    normal = "normal"
    double = "double"


class LLMTopping(BaseModel):
    name: str
    """Name of topping"""
    amount: Optional[str] = "normal"
    """Amount of topping"""

    def __str__(self):
        return f"{self.name.lower()}, {self.amount.lower()}"

    def to_order_item(self):
        """Convert to Topping"""
        try:
            return Topping(
                code=NAME_TO_TOPPING_CODE[self.name.lower()],
                amount=AMOUNT_TO_CODE[self.amount],
            )
        except KeyError:
            logging.warning(f"Invalid topping: {self.name}")
            return


DEFAULT_TOPPINGS = ["lettuce", "tomato", "onion", "ketchup"]


class LLMBurgerItem(BaseModel):
    name: str
    """Product code of item"""
    toppings: Optional[list[LLMTopping]] = None
    """Toppings of item"""
    size: Optional[str] = "quarter lb"
    """Size of item"""
    bun: Optional[str] = "sesame"
    """Bun of item"""
    patties: Optional[str] = "single"
    """Number of patties"""
    cook: Optional[str] = "normal"
    """Cook of item"""
    quantity: Optional[int] = 1
    """Quantity of item"""

    def __str__(self):
        return f"{self.name.lower()}, {self.size.lower()}, {self.patties.lower()}, {self.bun.lower()}, {self.cook.lower()}"

    @model_validator(mode="after")
    def validate_toppings(self):
        """Ensure default toppings are present and valid"""
        if self.toppings is None:
            self.toppings = [LLMTopping(name=n) for n in DEFAULT_TOPPINGS]
        else:
            # fill in missing default toppings
            current_toppings = [topping.name.lower() for topping in self.toppings]
            missing_toppings = [
                LLMTopping(name=n)
                for n in DEFAULT_TOPPINGS
                if n not in current_toppings
            ]
            self.toppings += missing_toppings

        # Only keep validated toppings
        valid_toppings = []
        for item in self.toppings:
            try:
                valid_item = item.to_order_item()
                if valid_item:
                    valid_toppings.append(item)
            except ValueError:
                logging.warning("Invalid topping: %s", str(item))
                continue
        self.toppings = valid_toppings
        return self

    @model_serializer
    def ser_model(self):
        """Serialize LLMBurgerItem to dictionary and remove default options"""
        model = {"name": self.name, "quantity": self.quantity}
        if self.size and self.size != "quarter lb":
            model["size"] = self.size
        if self.bun and self.bun != "sesame":
            model["bun"] = self.bun
        if self.patties and self.patties != "single":
            model["patties"] = self.patties
        if self.cook and self.cook != "normal":
            model["cook"] = self.cook
        if self.toppings:
            toppings = []
            for topping in self.toppings:
                if topping.amount != "normal" or topping.name not in DEFAULT_TOPPINGS:
                    toppings.append(topping.model_dump())
            if len(toppings) > 0:
                model["toppings"] = toppings
        return model

    def to_order_item(self):
        """Convert to Item"""
        toppings = []
        if self.toppings:
            for topping in self.toppings:
                try:
                    toppings.append(topping.to_order_item())
                except KeyError:
                    # Skip invalid toppings
                    continue
        try:
            return Item(
                code=NAME_TO_PRODUCT_CODE[str(self)],
                toppings=toppings,
                quantity=self.quantity,
            )
        except KeyError:
            logging.warning(f"Invalid burger item: {str(self)}")
            return


class LLMFriesItem(BaseModel):
    name: str
    """Name of item"""
    size: Optional[str] = "medium"
    """Size of item"""
    toppings: Optional[list[LLMTopping]] = [LLMTopping(name="salt", amount="normal")]
    """Salt option"""
    quantity: Optional[int] = 1
    """Quantity of item"""

    @model_validator(mode="after")
    def validate_toppings(self):
        """Ensure default salt topping"""
        v = self.toppings
        if v is None:
            self.toppings = [{"name": "salt", "amount": "normal"}]
        elif isinstance(v, list):
            # only allow salt
            toppings = [topping for topping in v if topping.name.lower() == "salt"]
            if len(toppings) != 0:
                self.toppings = [toppings[0]]
            else:
                self.toppings = [{"name": "salt", "amount": "normal"}]
        else:
            raise ValueError("Only salt topping is allowed for fries.")
        return self

    def __str__(self):
        if self.toppings and isinstance(self.toppings, list):
            salt_amount = self.toppings[0]
            if salt_amount.amount.lower() == "normal":
                salt = "salted"
            else:
                salt = "unsalted"
        else:
            salt = "salted"
        return f"{self.name.lower()}, {self.size.lower()}, {salt}"

    @model_serializer
    def ser_model(self):
        """Serialize LLMFriesItem to dictionary"""
        model = {"name": self.name, "quantity": self.quantity}
        if self.size and self.size != "medium":
            model["size"] = self.size
        if self.toppings and self.toppings[0].amount != "normal":
            model["amount"] = self.toppings[0].amount
        return model

    def to_order_item(self):
        """Convert to Item"""
        try:
            return Item(code=NAME_TO_PRODUCT_CODE[str(self)], quantity=self.quantity)
        except KeyError:
            logging.warning(f"Invalid fries item: {str(self)}")
            return


class LLMDrinkItem(BaseModel):
    name: str
    """Name of item"""
    size: Optional[str] = "medium"
    """Size of item"""
    quantity: Optional[int] = 1
    """Quantity of item"""

    def __str__(self):
        return f"{self.name.lower()}, {self.size.lower()}"

    @model_serializer
    def ser_model(self):
        """Serialize LLMDrinkItem to dictionary"""
        model = {"name": self.name, "quantity": self.quantity}
        if self.size and self.size != "medium":
            model["size"] = self.size
        return model

    def to_order_item(self):
        """Convert to Item"""
        try:
            return Item(code=NAME_TO_PRODUCT_CODE[str(self)], quantity=self.quantity)
        except KeyError:
            logging.warning(f"Invalid drink item: {str(self)}")
            return


def item_discriminator(item: Any) -> str:
    """Returns item type based on name

    Args:
        item (Any): dictionary or BaseModel to determine type

    Returns:
        str: item type
    """
    name = (item.get("name") if isinstance(item, dict) else item.name).lower()
    if "burger" in name:
        return "burger"
    elif "fries" in name:
        return "fries"
    return "drink"


LLMItem = Annotated[
    Union[
        Annotated[LLMFriesItem, Tag("fries")],
        Annotated[LLMDrinkItem, Tag("drink")],
        Annotated[LLMBurgerItem, Tag("burger")],
    ],
    Discriminator(item_discriminator),
]


class LLMOrder(BaseModel):
    items: list[LLMItem]

    @model_validator(mode="after")
    def validate_against_order(self):
        """Validate items against order schema and keep only valid items to ensure orders are synced"""
        valid_items = []
        for item in self.items:
            try:
                valid_item = item.to_order_item()
                if valid_item:
                    valid_items.append(item)
            except:
                continue
        self.items = valid_items
        return self

    @model_serializer
    def ser_model(self):
        """Serialize LLMOrder to dictionary"""
        return {"items": [item.model_dump() for item in self.items]}

    def to_order(self):
        """Convert LLMOrder to Order"""
        return Order(items=[item.to_order_item() for item in self.items])
