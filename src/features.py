from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import pandas as pd


def get_feature_list(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Возвращает списки числовых и категориальных признаков по договорённости.
    """
    numeric = [c for c in df.columns if c.startswith("num_")] + ["tenure"]
    categorical = ["plan", "contract"]
    return numeric, categorical


def build_preprocessor(numeric_features: list[str], categorical_features: list[str]) -> ColumnTransformer:
    """
    Возвращает ColumnTransformer:
      - StandardScaler для числовых;
      - OneHotEncoder с sparse=True для категорий.
    """
    num_pipeline = Pipeline([("scaler", StandardScaler())])
    cat_pipeline = Pipeline([("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=True))])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipeline, numeric_features),
            ("cat", cat_pipeline, categorical_features),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """
    Попытка получить имена признаков после трансформации.
    """
    feature_names = []
    for name, transformer, cols in preprocessor.transformers_:
        if name == "remainder":
            continue
        if hasattr(transformer, "named_steps") and "ohe" in transformer.named_steps:
            ohe = transformer.named_steps["ohe"]
            names = ohe.get_feature_names_out(cols)
            feature_names.extend(list(names))
        else:
            feature_names.extend(list(cols))
    return feature_names
