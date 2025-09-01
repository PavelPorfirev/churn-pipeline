# churn-pipeline

Этот репозиторий — небольшой учебный пример end-to-end ML-пайплайна.  
Здесь есть:

- генерация синтетических данных для задачи оттока клиентов;
- препроцессинг признаков через `ColumnTransformer`;
- модель на LightGBM с логированием метрик и артефактов в MLflow;
- гиперпараметрический поиск (Optuna) с трекингом в MLflow;
- инференс на FastAPI с валидацией входных данных;
- `docker-compose` для локального запуска MLflow;
- простые тесты и CI в GitHub Actions.

## Быстрый старт

1. Установка зависимостей:
```
python -m venv venv
source venv/bin/activate  # или venv\Scripts\activate на Windows
pip install -r requirements.txt
```

2. Поднять MLflow:
```
docker compose up -d mlflow
```
MLflow UI: http://localhost:5000

3. Обучить модель и логировать в MLflow:
```
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.train --output models/model.joblib
```

4. Запустить API:
```
export MODEL_PATH=models/model.joblib
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

5. Запустить HPO (Optuna):
```
export MLFLOW_TRACKING_URI=http://localhost:5000
python -m src.hpo --trials 40 --save-best best_params.json
```

6. Анализ результатов:
```
python scripts/visualize_optuna.py --out-folder analysis
```
