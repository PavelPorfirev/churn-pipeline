from sklearn.datasets import make_classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def generate_synthetic_churn(
    n_samples: int = 20_000, random_state: int = 42
) -> pd.DataFrame:
    """
    Генерирует синтетический табличный датасет для задачи оттока.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=12,
        n_informative=6,
        n_redundant=2,
        weights=[0.8, 0.2],
        flip_y=0.01,
        class_sep=1.0,
        random_state=random_state,
    )
    df = pd.DataFrame(X, columns=[f"num_{i}" for i in range(X.shape[1])])
    rng = np.random.RandomState(random_state)
    df["tenure"] = np.clip(rng.exponential(scale=12, size=n_samples).astype(int), 0, 72)
    df["plan"] = rng.choice(["basic", "plus", "pro"], size=n_samples, p=[0.6, 0.3, 0.1])
    df["contract"] = rng.choice(
        ["month-to-month", "one-year", "two-year"], size=n_samples, p=[0.5, 0.3, 0.2]
    )
    df["target"] = y
    return df


def load_split(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Разбивает DataFrame на X_train, X_test, y_train, y_test со стратификацией по целевой переменной.
    """
    X = df.drop(columns=["target"])
    y = df["target"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    return (
        X_train.reset_index(drop=True),
        X_test.reset_index(drop=True),
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
    )
