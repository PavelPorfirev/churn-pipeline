from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from joblib import load
import pandas as pd
import os
import logging
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = os.environ.get("MODEL_PATH", "models/model.joblib")
METADATA_PATH = os.environ.get("MODEL_METADATA_PATH", MODEL_PATH + ".metadata.json")


class Instances(BaseModel):
    instances: list[dict[str, object]]


app = FastAPI(title="Churn model", version="0.2")
model = None
expected_columns: list[str] | None = None


@app.on_event("startup")
def load_model():
    """
    Загружает модель и метаданные при старте приложения.
    """
    global model, expected_columns
    if not os.path.exists(MODEL_PATH):
        logger.error("Модель не найдена по пути: %s", MODEL_PATH)
        raise RuntimeError("Модель не найдена. Запустите обучение и задайте path.")
    model = load(MODEL_PATH)
    logger.info("Модель загружена из %s", MODEL_PATH)

    if os.path.exists(METADATA_PATH):
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                md = json.load(f)
            expected_columns = md.get("feature_columns")
            logger.info("Загружено %d ожидаемых колонок", len(expected_columns) if expected_columns else 0)
        except Exception as e:
            logger.warning("Не удалось загрузить метаданные %s: %s", METADATA_PATH, e)
            expected_columns = None
    else:
        expected_columns = None
        logger.info(
            "Файл метаданных не найден: %s — вход будет проходить без явной валидации колонок",
            METADATA_PATH,
        )


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: Instances):
    """
    Ожидает JSON: {"instances": [ {feature: value, ...}, ... ]}
    Возвращает список вероятностей оттока.
    """
    try:
        X = pd.DataFrame(payload.instances)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Неверный формат instances: {e}")

    # Если есть метаданные с ожидаемыми колонками — приводим DataFrame к этому виду
    if expected_columns:
        missing = [c for c in expected_columns if c not in X.columns]
        if missing:
            for c in missing:
                X[c] = 0
            logger.info("Добавлены отсутствующие колонки: %s", missing)

        extra = [c for c in X.columns if c not in expected_columns]
        if extra:
            X = X.drop(columns=extra)
            logger.info("Удалены лишние колонки: %s", extra)

        X = X[expected_columns]

    try:
        probs = model.predict_proba(X)[:, 1].tolist()
    except Exception as e:
        logger.exception("Ошибка инференса модели")
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {e}")

    return {"predictions": probs}
