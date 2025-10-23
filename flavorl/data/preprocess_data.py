import os
import re
import ast
import json
import pandas as pd
from pathlib import Path
from flavorl.data.data_utils import extract_times_and_clean, _clean_and_unpack

# 1) RUTAS
DATA_DIR = Path("MealRecPlus/MealRec+/MealRec+H")
meta_dir = DATA_DIR / "meta_data"
health_dir = DATA_DIR / "healthiness"
rel_dir = DATA_DIR  # los .txt de relaciones están en la raíz de MealRec+H

# Rutas directas (sin coalesce_path)
course_csv_fp = meta_dir / "course.csv"  # cuidado que hay que hacer unzip
user_course_csv_fp = meta_dir / "user_course.csv"
user2index_fp = meta_dir / "user2index.txt"
course2index_fp = meta_dir / "course2index.txt"

user_course_txt_fp = rel_dir / "user_course.txt"
course_category_fp = rel_dir / "course_category.txt"
meal_course_fp = rel_dir / "meal_course.txt"
user_meal_fp = rel_dir / "user_meal.txt"  # opcional

# splits de user_meal:
user_meal_train_fp = rel_dir / "user_meal_train.txt"
user_meal_tune_fp = rel_dir / "user_meal_tune.txt"
user_meal_test_fp = rel_dir / "user_meal_test.txt"

course_fsa_fp = health_dir / "course_fsa.txt"
course_who_fp = health_dir / "course_who.txt"
meal_fsa_fp = health_dir / "meal_fsa.txt"
meal_who_fp = health_dir / "meal_who.txt"
user_fsa_fp = health_dir / "user_fsa.txt"
user_who_fp = health_dir / "user_who.txt"

# Comprobaciones
required_paths = [
    course_csv_fp,
    user_course_csv_fp,
    user2index_fp,
    course2index_fp,
    user_course_txt_fp,
    course_category_fp,
    meal_course_fp,
    course_fsa_fp,
    course_who_fp,
    meal_fsa_fp,
    meal_who_fp,
    user_fsa_fp,
    user_who_fp,
]
missing = [str(p) for p in required_paths if not p.exists()]
if missing:
    raise FileNotFoundError("Faltan estos archivos:\n  - " + "\n  - ".join(missing))


# ==============================
# 2) META COURSE
# ==============================
df_courses = pd.read_csv(course_csv_fp)

print(df_courses.shape)
print(df_courses.columns)
print(df_courses.head())


import ast
import pandas as pd

df_courses["cooking_directions_original"] = df_courses["cooking_directions"]
df_courses[["cooking_directions", "prep_min", "cook_min", "ready_min"]] = df_courses[
    "cooking_directions"
].apply(_clean_and_unpack)

print(df_courses.head())


course2index_fp = "MealRecPlus/MealRec+/MealRec+H/meta_data/course2index.txt"
course_fsa_fp = "MealRecPlus/MealRec+/MealRec+H/healthiness/course_fsa.txt"
course_who_fp = "MealRecPlus/MealRec+/MealRec+H/healthiness/course_who.txt"

# --- 1) course_id -> course_index ---
df_c2i = pd.read_csv(
    course2index_fp, sep="\t", header=None, names=["course_id", "course_index"]
)

# If your main dataframe is df_courses (has column 'course_id'):
df_courses = df_courses.merge(df_c2i, on="course_id", how="left")

# Optional: make it an integer (nullable) so NaNs are allowed
df_courses["course_index"] = df_courses["course_index"].astype("Int64")

# --- 2) load FSA/WHO series (line index == course_index) ---
# Each line is a value; its row number is the course_index.
s_fsa = pd.read_csv(course_fsa_fp, header=None).squeeze("columns")
s_who = pd.read_csv(course_who_fp, header=None).squeeze("columns")

s_fsa.index.name = "course_index"
s_fsa.name = "course_fsa"
s_who.index.name = "course_index"
s_who.name = "course_who"

# --- 3) join by course_index ---
df_courses = df_courses.join(s_fsa, on="course_index")
df_courses = df_courses.join(s_who, on="course_index")

# Optional sanity checks
print("Unmapped course_id count:", df_courses["course_index"].isna().sum())
print(
    "Missing FSA:",
    df_courses["course_fsa"].isna().sum(),
    "Missing WHO:",
    df_courses["course_who"].isna().sum(),
)
