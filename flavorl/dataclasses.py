from enum import Enum
from dataclasses import dataclass
from typing import Any, List, Type, TypeVar

import polars as pl
import random

T = TypeVar("T")

class MealType(Enum):
    BREAKFAST = 0
    LUNCH = 1
    DINNER = 2
    
class Day(Enum):
    MONDAY = 0
    TUESDAY = 1
    WEDNESDAY = 2
    THURSDAY = 3
    FRIDAY = 4
    SATURDAY = 5
    SUNDAY = 6


@dataclass
class User:
    """
    Represents a single user entry.

    Attributes:
        user_idx (int): Unique identifier for the user.
        allergies (dict[bool]): dict indicating known allergies.
        intoler (dict[bool]): dict indicating the user's food intolerances.
        vegan (bool): Whether the user follows a vegan diet.
        vegetarian (bool): Whether the user follows a vegetarian diet.
        preferences (str): Description of the user's dietary preferences or tastes.
        daily_cal (float): Remaining daily calories to be consumed.
        daily_nutr (dict[str, float]): Remaining daily nutrients to be consumed.
    """

    user_idx: int

    allergies: dict[str, bool]
    intoler: dict[str, bool]
    vegan: bool
    vegetarian: bool
    preferences: str
    daily_cal: float
    daily_nutr: dict[str, float]

@dataclass
class Meal:
    """
    Represents a single meal item with its nutritional information.

    Attributes:
        meal_idx (int): Unique identifier for the meal.
        meal_type (int): Integer representing the meal type (e.g., 0 breakfast, 1 lunch, etc.).
        calories (float): Total caloric content of the meal.
        nutrients (dict[str, float]): dictionary mapping nutrient names to their corresponding quantities.
        ingredients (str): Description or list of the ingredients used in the meal.
        tags (str): Descriptive tags or labels associated with the meal.
        healthy_score (float): Aggregated healthy scores (course_fsa + course_who).
    """

    meal_idx: int

    meal_type: int
    calories: float
    nutrients: dict[str, float]
    ingredients: str
    tags: str
    healthy_score : float


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
            **filters: Column filters.

        Returns:
            List[T]: Sampled objects as instances of the dataclass.
        """
        df = self.df
        for col, val in filters.items():
            if col in df.columns:
                df = df.filter(pl.col(col) == val)

        if df.is_empty():
            return []

        rows: List[dict[str, Any]] = df.to_dicts()
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
