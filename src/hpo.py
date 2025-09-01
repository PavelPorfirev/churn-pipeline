import argparse
import json
import logging
import optuna
from optuna.integration.mlflow import MLflowCallback
import mlflow
from sklearn.pipeline import Pipeline
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from .data import generate_synthetic_churn, load_split
from .features import build_preprocessor, get_feature_list

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def objective(trial: optuna.Trial) -> float:
    """
    Целевая функция Optuna: строит pipeline и возвращает AUC-ROC на валидации.
    """
    params: dict[str, object] = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 600),
        "learning_rate": trial.suggest_loguniform("learning_rate", 1e-3, 0.3),
        "num_leaves": trial.suggest_int("num_leaves", 16, 256),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
        "subsample": trial.suggest_float("subsample", 0.4, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
    }

    df = generate_synthetic_churn(n_samples=8000, random_state=0)
    X_train, X_test, y_train, y_test = load_split(df, test_size=0.2, random_state=0)

    numeric, categorical = get_feature_list(X_train)
    preprocessor = build_preprocessor(numeric, categorical)

    model = LGBMClassifier(**params, random_state=0, n_jobs=-1)
    pipeline = Pipeline([("preprocessor", preprocessor), ("clf", model)])
    pipeline.fit(X_train, y_train)
    probs = pipeline.predict_proba(X_test)[:, 1]
    roc = float(roc_auc_score(y_test, probs))
    return roc


def run_hpo(n_trials: int = 40, study_name: str = "churn-optuna", save_best: str | None = None):
    """
    Запускает исследование Optuna, логирует trials в MLflow и сохраняет лучшие параметры.
    """
    mlflow_uri = mlflow.get_tracking_uri()
    logger.info("MLflow tracking URI: %s", mlflow_uri)

    mlflow_cb = MLflowCallback(tracking_uri=mlflow_uri, metric_name="roc_auc")
    study = optuna.create_study(direction="maximize", study_name=study_name)
    study.optimize(objective, n_trials=n_trials, callbacks=[mlflow_cb])

    logger.info("Лучшее значение: %s", study.best_value)
    logger.info("Лучшие параметры: %s", study.best_trial.params)

    if save_best:
        with open(save_best, "w", encoding="utf-8") as f:
            json.dump({"best_value": study.best_value, "best_params": study.best_trial.params}, f, indent=2)
        logger.info("Сохранено в %s", save_best)

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=40)
    parser.add_argument("--study-name", type=str, default="churn-optuna")
    parser.add_argument("--save-best", type=str, default="best_params.json")
    args = parser.parse_args()
    run_hpo(n_trials=args.trials, study_name=args.study_name, save_best=args.save_best)
