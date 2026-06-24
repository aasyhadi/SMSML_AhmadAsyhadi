from pathlib import Path
import mlflow
mlflow.set_tracking_uri("file:./mlruns")
exp = mlflow.get_experiment_by_name("SMSML Breast Cancer - AhmadAsyhadi")
runs = mlflow.search_runs(experiment_ids=[exp.experiment_id]) if exp else None
run_id = runs.iloc[0]["run_id"] if runs is not None and not runs.empty else "RUN_ID_BELUM_TERSEDIA"
Path("latest_run_id.txt").write_text(run_id)
print(run_id)
