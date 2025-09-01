import os
import argparse
import mlflow
import pandas as pd
import matplotlib.pyplot as plt


def ensure_dir(p):
    if not os.path.exists(p):
        os.makedirs(p)


def load_mlflow_runs(experiment_name: str | None = None) -> pd.DataFrame:
    """
    Получает DataFrame со всеми запусками (или для указанного эксперимента).
    """
    if experiment_name:
        exp = mlflow.get_experiment_by_name(experiment_name)
        if exp is None:
            raise RuntimeError(f"Эксперимент {experiment_name} не найден")
        df = mlflow.search_runs(
            experiments=[exp.experiment_id], output_format="dataframe"
        )
    else:
        df = mlflow.search_runs(output_format="dataframe")
    return df


def main(out_folder="analysis", experiment_name=None):
    ensure_dir(out_folder)
    df = load_mlflow_runs(experiment_name)
    if df.empty:
        print("Нет запусков в MLflow.")
        return

    # Попытка найти метрику roc_auc среди колонок metrics.*
    metric_col = next(
        (c for c in df.columns if c.startswith("metrics.") and "roc_auc" in c), None
    )

    if metric_col:
        plt.figure()
        df_sorted = df.sort_values("start_time")
        plt.plot(
            df_sorted["start_time"].astype("int64") // 10**9, df_sorted[metric_col]
        )
        plt.xlabel("start_time (s since epoch)")
        plt.ylabel(metric_col)
        plt.title("Метрика по времени (по порядку trial)")
        plt.tight_layout()
        plt.savefig(os.path.join(out_folder, "metric_over_time.png"))
        print("Сохранён metric_over_time.png")

    # Сохраняем сводную таблицу параметров/метрик
    params = [c for c in df.columns if c.startswith("params.")]
    metrics = [c for c in df.columns if c.startswith("metrics.")]
    selected = (
        ["run_id", "status", "start_time"]
        + params
        + ([metric_col] if metric_col else metrics)
    )
    df[selected].to_csv(os.path.join(out_folder, "runs_summary.csv"), index=False)
    print("Сохранён runs_summary.csv в", out_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-folder", default="analysis")
    parser.add_argument("--experiment-name", default=None)
    args = parser.parse_args()
    main(out_folder=args.out_folder, experiment_name=args.experiment_name)
