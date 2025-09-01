import os
import argparse
import mlflow
import mlflow.sklearn
from joblib import dump
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
from .data import generate_synthetic_churn, load_split
from .features import build_preprocessor, get_feature_list
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def train(
    output: str = "models/model.joblib",
    params: dict[str, object] | None = None,
    random_state: int = 42,
):
    """
    Обучает пайплайн и сохраняет его в файл и в MLflow.
    """
    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    params = params or {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31}

    df = generate_synthetic_churn(n_samples=20_000, random_state=random_state)
    X_train, X_test, y_train, y_test = load_split(
        df, test_size=0.2, random_state=random_state
    )

    numeric, categorical = get_feature_list(X_train)
    preprocessor = build_preprocessor(numeric, categorical)

    model = LGBMClassifier(**params, random_state=random_state, n_jobs=-1)
    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", model)])

    with mlflow.start_run():
        pipeline.fit(X_train, y_train)
        probs = pipeline.predict_proba(X_test)[:, 1]
        roc = float(roc_auc_score(y_test, probs))
        ap = float(average_precision_score(y_test, probs))
        mlflow.log_params(params)
        mlflow.log_metrics({"roc_auc": roc, "average_precision": ap})
        mlflow.sklearn.log_model(pipeline, "model")

    # сохраняем модель на диск
    dump(pipeline, output)
    logger.info("Пайплайн сохранён в %s", output)

    # сохраняем метаданные с набором исходных колонок
    feature_columns = X_train.columns.tolist()
    metadata_path = f"{output}.metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as mf:
        json.dump(
            {"feature_columns": feature_columns}, mf, ensure_ascii=False, indent=2
        )
    logger.info("Метаданные сохранены в %s", metadata_path)

    return {"roc_auc": roc, "average_precision": ap}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="models/model.joblib")
    args = parser.parse_args()
    print(train(output=args.output))
