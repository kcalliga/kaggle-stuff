import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

# Basic pandas / plotting defaults
pd.set_option("display.max_rows", 100)
pd.set_option("display.max_columns", 100)
sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["figure.dpi"] = 100


# ==========================
# Core Data Utilities
# ==========================

def load_csv(
    path: Union[str, Path],
    nrows: Optional[int] = None,
    **read_csv_kwargs,
) -> pd.DataFrame:
    # Load a CSV from a path into a DataFrame with a small convenience wrapper.
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")
    df = pd.read_csv(path, nrows=nrows, **read_csv_kwargs)
    print(f"Loaded {path} with shape {df.shape}")
    return df


def summarize_dataframe(df: pd.DataFrame, name: str = "df") -> None:
    # Quick summary: head, dtypes, missing values.
    print(f"===== Summary: {name} =====")
    print("Shape:", df.shape)
    display(df.head())
    print("\nDtypes:")
    display(df.dtypes)
    print("\nMissing (%) by column:")
    display((df.isna().mean() * 100).sort_values(ascending=False))


def get_numeric_features(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    # Return list of numeric column names, optionally excluding some columns.
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if exclude:
        cols = [c for c in cols if c not in exclude]
    return cols


def get_categorical_features(df: pd.DataFrame, exclude: Optional[List[str]] = None) -> List[str]:
    # Return list of categorical / object column names, optionally excluding some columns.
    cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
    if exclude:
        cols = [c for c in cols if c not in exclude]
    return cols


def basic_train_valid_split(
    df: pd.DataFrame,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Convenience wrapper around train_test_split.
    X = df.drop(columns=[target_col])
    y = df[target_col]

    stratify_vals = y if stratify else None

    X_train, X_valid, y_train, y_valid = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify_vals
    )

    print(f"Train shape: {X_train.shape}, Valid shape: {X_valid.shape}")
    return X_train, X_valid, y_train, y_valid


# ==========================
# EDA Plotting Helpers
# ==========================

def plot_numeric_distributions(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    bins: int = 30,
    max_cols: int = 12,
    title: Optional[str] = None,
) -> None:
    # Plot histograms for numeric columns (or subset).
    if cols is None:
        cols = get_numeric_features(df)
    cols = cols[:max_cols]  # avoid too many subplots

    df[cols].hist(bins=bins, figsize=(4 * len(cols), 4))
    if title:
        plt.suptitle(title, y=1.02)
    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: Optional[List[str]] = None,
    max_cols: int = 30,
    title: str = "Correlation Heatmap",
) -> None:
    # Plot correlation heatmap for numeric columns (or subset).
    if cols is None:
        cols = get_numeric_features(df)
    cols = cols[:max_cols]
    corr = df[cols].corr()
    sns.heatmap(corr, cmap="coolwarm", center=0)
    plt.title(title)
    plt.show()


def plot_target_distribution(
    y: Union[pd.Series, np.ndarray],
    title: str = "Target distribution",
) -> None:
    # Quick barplot/histogram for target variable (classification or regression).
    s = pd.Series(y)
    if s.dtype.kind in "ifu":
        sns.histplot(s, bins=30, kde=False)
    else:
        s.value_counts().plot(kind="bar")
    plt.title(title)
    plt.show()
