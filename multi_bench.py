import os
import re
import sys
from pathlib import Path

import pandas as pd
import wandb
from datasets import load_dataset
from zeus.monitor import ZeusMonitor

from routellm.controller import Controller


# ─────────────────────────────────────────────────────────
# Configuration and paths
# ─────────────────────────────────────────────────────────
parent_folder = Path(__file__).parent.resolve()
results_root = parent_folder / "sweep_results"
results_root.mkdir(exist_ok=True)

# --- one-liner style dictionary -------------------------------------------
DATASET_CONFIGS = {
    "boolq":        {"path":"boolq","split":"validation","question":"question","label":"answer"},
    "arc_easy":     {"path":"ai2_arc","config":"ARC-Easy","split":"test","question":"question","label":"answerKey"},
    "arc_challenge":{"path":"ai2_arc","config":"ARC-Challenge","split":"test","question":"question","label":"answerKey"},
    "gsm8k":        {"path":"gsm8k","config":"main","split":"test","question":"question","label":"answer"},
    "sciq":         {"path":"sciq","split":"validation","question":"question","label":"correct_answer"},
    "piqa":         {"path":"piqa","split":"test","question":"goal","label":"label"},
    "logiqa":       {"path":"lucasmccabe/logiqa","split":"validation","context":"context","question":"query","label":"correct_option"},
    "logiqa2":      {"path":"baber/logiqa2","config":"logiqa2","split":"validation","question":"question","label":"answer"},
    "social_iqa":   {"path":"social_i_qa","split":"validation","question":"question","label":"label"},
    "winogrande":   {"path":"winogrande","config":"winogrande_xl","split":"validation","question":"sentence","label":"answer"},
}

BENCHMARKS   = list(DATASET_CONFIGS.keys())
ROUTER       = "bert"
THRESHOLDS   = [0.05, 0.20, 0.45, 0.70]

# optional HF token
api_key_path = os.environ.get("HF_TOKEN_PATH", "")
api_key      = Path(api_key_path).read_text().strip() if api_key_path else ""

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)


# ─────────────────────────────────────────────────────────
# Dataset loader
# ─────────────────────────────────────────────────────────
def load_task_dataframe(task_name: str) -> pd.DataFrame:
    cfg = DATASET_CONFIGS[task_name]
    ds  = load_dataset(cfg["path"], cfg.get("config"), split=cfg["split"])

    # --- build question text ---------------------------------------------
    if task_name == "logiqa":
        questions = [
            f"{ctx.strip()}\n\n{q.strip()}"
            for ctx, q in zip(ds[cfg["context"]], ds[cfg["question"]])
        ]
    else:
        questions = ds[cfg["question"]]

    # --- canonicalise gold label -----------------------------------------
    if task_name.startswith("arc_"):
        labels = [
            row_choices["label"].index(ans_key)
            for ans_key, row_choices in zip(ds[cfg["label"]], ds["choices"])
        ]

    elif task_name == "social_iqa":
        labels = [ord(ch) - ord("A") for ch in ds[cfg["label"]]]

    else:
        labels = ds[cfg["label"]]

    return pd.DataFrame({"question": questions, "label": labels})


# ─────────────────────────────────────────────────────────
# Inference helper
# ─────────────────────────────────────────────────────────
def get_completion(row, client, thr):
    prompt    = f"Question: {row['question']}\nRespond with ONLY 'true' or 'false':"
    messages  = [{"role": "user", "content": prompt}]
    model_id  = f"router-{ROUTER}-{thr}"
    responses, routed_model = client.chat.completions.create(model=model_id, messages=messages)

    raw  = responses[0].outputs[0].text.strip().lower()
    m    = re.search(r"\b(true|false)\b", raw)
    pred = (m.group(0) == "true") if m else None
    return pred, raw, routed_model


# ─────────────────────────────────────────────────────────
# Main sweep for one benchmark
# ─────────────────────────────────────────────────────────
def main(task_idx: int):
    task_name = BENCHMARKS[task_idx]
    wandb.login()

    client = Controller(
        routers=[ROUTER],
        strong_model="meta-llama/Llama-3.1-8B-Instruct",
        weak_model="meta-llama/Llama-3.2-1B-Instruct",
        api_key=api_key,
        progress_bar=True,
    )

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

        monitor = ZeusMonitor(gpu_indices=[0], approx_instant_energy=True)
        records = []

        for idx, row in df_base.iterrows():
            monitor.begin_window(f"sample-{idx}")
            pred, raw, model = get_completion(row, client, thr)
            meas   = monitor.end_window(f"sample-{idx}")
            energy = sum(meas.gpu_energy.values())
            correct     = (pred == row["label"])
            choice_int  = 1 if "8B" in model else 0

            wandb.log(
                {
                    "sample_energy": energy,
                    "sample_accuracy": int(correct),
                    "sample_model_choice": choice_int,
                },
                step=idx,
                commit=True,
            )

            print(
                f"[{task_name} thr={thr:.2f} | sample {idx}] "
                f"energy={energy:.2f} J, correct={correct}, model={model}"
            )

            records.append(
                {
                    "question":  row["question"],
                    "label":     row["label"],
                    "predicted": pred,
                    "raw":       raw,
                    "model":     model,
                    "energy":    energy,
                    "correct":   correct,
                    "choice_int": choice_int,
                }
            )

        df            = pd.DataFrame(records)
        acc           = df.correct.mean()
        total_energy  = df.energy.sum()
        avg_choice    = df.choice_int.mean()

        wandb.log({"accuracy": acc, "energy_j": total_energy, "model_choice": avg_choice})

        out_dir = results_root / task_name
        out_dir.mkdir(exist_ok=True)
        df.to_csv(out_dir / f"{task_name}_{thr:.2f}.csv", index=False)

        print(
            f"**DONE** {task_name} thr={thr:.2f} "
            f"acc={acc:.3%} energy={total_energy:.1f} J choice={avg_choice:.2f}"
        )
        run.finish()


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        print(
            f"Usage: python {sys.argv[0]} <benchmark_idx 0..{len(BENCHMARKS)-1}> "
            "[optional: single_threshold]"
        )
        sys.exit(1)

    if len(sys.argv) == 3:               
        THRESHOLDS = [float(sys.argv[2])]
    
    main(int(sys.argv[1]))