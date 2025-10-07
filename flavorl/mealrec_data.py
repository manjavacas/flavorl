from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type, TypeVar
import polars as pl
import random

T = TypeVar("T")


@dataclass
class User:
    """
    Represents a single user entry.

    Attributes:
        user_id (str): Unique identifier of the user.
        dietary_preferences (str): Dietary preferences (e.g., vegan, vegetarian, omnivore).
        allergies (str): Allergies or food restrictions.
    """

    user_id: str

    # --- TODO: define additional user info ---
    dietary_preferences: str
    allergies: str


@dataclass
class Meal:
    """
    Represents a single meal item with its nutritional information.

    Attributes:
        name (str): Name of the meal.
        type (str): Type or category of the meal (e.g., breakfast, lunch, snack).
        calories (int, optional): Total calories in the meal.
        protein_g (float): Protein content in grams.
        fat_g (float): Fat content in grams.
        carbs_g (float): Carbohydrate content in grams.
        fiber_g (float): Fiber content in grams.
        sodium_mg (float): Sodium content in milligrams.
    """

    name: str
    type: str

    # --- TODO: define additional meal info ---
    calories: int
    protein_g: float
    fat_g: float
    carbs_g: float
    fiber_g: float
    sodium_mg: float


class BaseDataset:
    """
    Generic CSV-backed dataset.

    Provides filtering and sampling utilities.
    """

    def __init__(self, csv_file: str, dataclass_type: Type[T]) -> None:
        """
        Initializes the dataset from a CSV file.

        Args:
            csv_file (str): Path to the CSV file.
            dataclass_type (Type[T]): Dataclass type for the rows (Meal, User, etc.)
        """
        self.df: pl.DataFrame = pl.read_csv(csv_file)
        self.dataclass_type = dataclass_type

    def sample(self, n: int = 1, **filters: Any) -> List[T]:
        """
        Returns a random sample of objects from the dataset, optionally filtered.

        Args:
            n (int): Number of items to sample.
            **filters: Column filters, e.g., type="fruit" or gender="female".

        Returns:
            List[T]: Sampled objects as instances of the dataclass.
        """
        df = self.df
        for col, val in filters.items():
            if col in df.columns:
                df = df.filter(pl.col(col) == val)

        if df.is_empty():
            return []

        rows: List[Dict[str, Any]] = df.to_dicts()
        if n <= len(rows):
            sampled_rows = random.sample(rows, n)
        else:
            sampled_rows = [random.choice(rows) for _ in range(n)]

        return [self.dataclass_type(**row) for row in sampled_rows]


class MealDataset(BaseDataset):
    """
    Dataset for meals.
    """

    def __init__(self, csv_file: str):
        super().__init__(csv_file, Meal)


class UserDataset(BaseDataset):
    """
    Dataset for users.
    """

    def __init__(self, csv_file: str):
        super().__init__(csv_file, User)
