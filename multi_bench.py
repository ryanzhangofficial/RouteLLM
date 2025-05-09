import os
import re
from pathlib import Path

import pandas as pd
import wandb
from datasets import load_dataset
from zeus.monitor import ZeusMonitor

from routellm.controller import Controller

# ──────────────────────────────────────────────────────────────────────────
# Configuration and paths
# ──────────────────────────────────────────────────────────────────────────
parent_folder = Path(__file__).parent.resolve()
results_root  = parent_folder / "sweep_results"
results_root.mkdir(exist_ok=True)

# Define benchmarks and how to load them via 🤗 datasets
DATASET_CONFIGS = {
    "boolq": {
        "path": "boolq", "split": "validation",
        "question": "question", "label": "answer"
    },
    "arc_easy": {
        "path": "ai2_arc", "config": "ARC-Easy", "split": "test",
        "question": "question", "label": "answerKey"
    },
    "arc_challenge": {
        "path": "ai2_arc", "config": "ARC-Challenge", "split": "test",
        "question": "question", "label": "answerKey"
    },
    "gsm8k": {
        "path": "gsm8k", "split": "test",
        "question": "question", "label": "answer"
    },
    "sciq": {
        "path": "scitail", "split": "test",
        "question": "premise", "label": "hypothesis_label"
    },
    # add more datasets as needed
}

BENCHMARKS = list(DATASET_CONFIGS.keys())
ROUTER     = "bert"
THRESHOLDS = [0.05, 0.20, 0.45, 0.70]

# Read HF API key from mounted file
api_key_path = os.environ.get("HF_TOKEN_PATH", "")
api_key      = Path(api_key_path).read_text().strip() if api_key_path else ""

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# ──────────────────────────────────────────────────────────────────────────
# Dataset loader
# ──────────────────────────────────────────────────────────────────────────

def load_task_dataframe(task_name: str) -> pd.DataFrame:
    cfg = DATASET_CONFIGS.get(task_name)
    if cfg is None:
        raise ValueError(f"Unknown benchmark: {task_name}")
    path   = cfg["path"]
    split  = cfg["split"]
    config = cfg.get("config", None)
    ds = load_dataset(path, config, split=split)
    q_key = cfg["question"]
    l_key = cfg["label"]
    questions = ds[q_key]
    labels     = ds[l_key]
    # For ARC tasks 'answerKey' is e.g. 'A','B','C','D'; map to index
    if task_name.startswith("arc_"):
        choice_map = {c: i for i, c in enumerate(ds["choices"]["label"])}
        labels = [choice_map[k] for k in labels]
    return pd.DataFrame({"question": questions, "label": labels})

# ──────────────────────────────────────────────────────────────────────────
# Inference helper
# ──────────────────────────────────────────────────────────────────────────

def get_completion(row, client, thr):
    prompt = f"Question: {row['question']}\nRespond with ONLY 'true' or 'false':"
    messages = [{"role": "user", "content": prompt}]
    model_id = f"router-{ROUTER}-{thr}"
    responses, routed_model = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )
    raw = responses[0].outputs[0].text.strip().lower()
    m   = re.search(r"\b(true|false)\b", raw)
    pred = (m.group(0) == "true") if m else None
    return pred, raw, routed_model

# ──────────────────────────────────────────────────────────────────────────
# Main sweep with Zeus energy monitor
# ──────────────────────────────────────────────────────────────────────────

def main():
    wandb.login()
    client = Controller(
        routers=[ROUTER],
        strong_model="meta-llama/Llama-3.1-8B-Instruct",
        weak_model="meta-llama/Llama-3.2-1B-Instruct",
        api_key=api_key,
        progress_bar=True,
    )

    for task_name in BENCHMARKS:
        df_base = load_task_dataframe(task_name)
        print(f"Loaded {task_name}: {len(df_base)} samples")

        for thr in THRESHOLDS:
            run = wandb.init(
                project="routellm-sweep",
                entity="tum-i13",
                name=f"{ROUTER}-{task_name}-thr-{thr:.2f}",
                config={"benchmark": task_name, "router": ROUTER, "threshold": thr},
                reinit=True,
            )

            # Initialize Zeus monitor
            monitor = ZeusMonitor(gpu_indices=[0], approx_instant_energy=True)
            monitor.begin_window(f"{task_name}-thr-{thr}")

            # Inference
            df = df_base.copy()
            df[["predicted", "raw", "model"]] = df.apply(
                lambda r: pd.Series(get_completion(r, client, thr)), axis=1
            )

            # End Zeus window and sum energy
            measurement = monitor.end_window(f"{task_name}-thr-{thr}")
            energy_j = sum(measurement.gpu_energy.values())

            # Metrics
            df["correct"] = df.predicted == df.label
            acc = df.correct.mean()
            df["choice_int"] = df.model.apply(lambda m: 1 if "8B" in m else 0)
            avg_choice = df.choice_int.mean()

            # Log
            run.log({"accuracy": acc, "energy_j": energy_j, "model_choice": avg_choice})

            # Save CSV
            out_dir = results_root / task_name
            out_dir.mkdir(exist_ok=True)
            df.to_csv(out_dir / f"{task_name}_{thr:.2f}.csv", index=False)

            print(f"{task_name} thr={thr:.2f} acc={acc:.3%} energy={energy_j:.1f}J choice={avg_choice:.2f}")
            run.finish()

if __name__ == "__main__":
    main()