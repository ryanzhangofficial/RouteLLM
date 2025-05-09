import os, re, time
from pathlib import Path

import pandas as pd
import wandb
from zeus.energy_meter import EnergyMeter
from lm_eval.tasks import TASK_REGISTRY

from routellm.controller import Controller

# ──────────────────────────────────────────────────────────────────────────
# 0) Path helpers
# ──────────────────────────────────────────────────────────────────────────
parent_folder = Path(__file__).parent.absolute()       
results_root  = parent_folder / "sweep_results"
results_root.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────
# 1) Globals
# ──────────────────────────────────────────────────────────────────────────
BENCHMARKS   = [                     
    "arc_challenge",
    "arc_easy",
    "boolq",
    "lambada_standard",
    "logiqa",
    "logiqa2",
    "piqa",
    "sciq",
    "social_iqa",
    "winogrande",
]
ROUTER = "bert"
THRESHOLDS = [0.05, 0.20, 0.45, 0.70]

api_key_path = os.environ["HF_TOKEN_PATH"]
api_key      = Path(api_key_path).read_text().strip()

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

# ──────────────────────────────────────────────────────────────────────────
# 2) Helpers
# ──────────────────────────────────────────────────────────────────────────
def load_task_dataframe(task_name: str) -> pd.DataFrame:
    """Return a DataFrame of (question, label) using lm-eval’s task API."""
    task = TASK_REGISTRY.get(task_name)()
    docs = task.test_docs() if hasattr(task, "test_docs") else task.validation_docs()
    qs, ls = [], []

    for d in docs:
        qs.append(task.doc_to_text(d))
        ls.append(task.doc_to_target(d))

    return pd.DataFrame({"question": qs, "label": ls})


def zeus_energy(fn, *args, **kwargs):
    """Run fn(*args, **kwargs) and return (result, joules)."""
    meter = EnergyMeter()
    meter.start()
    res = fn(*args, **kwargs)
    meter.stop()
    return res, meter.get_total_energy()


def model_choice_to_int(name: str) -> int:
    """Strong model returns 1, weak returns 0 for W&B graph."""
    # crude rule: 8B in name ⇒ strong
    return 1 if "8B" in name or "8b" in name else 0


def get_completion(row, client, thr):
    prompt = f"Question: {row['question']}\nRespond with ONLY 'true' or 'false':\n"
    msgs   = [{"role": "user", "content": prompt}]
    model_id = f"router-{ROUTER}-{thr}"

    responses, routed_model = client.chat.completions.create(
        model=model_id, messages=msgs
    )

    raw  = responses[0].outputs[0].text.strip().lower()
    m    = re.search(r"\b(true|false)\b", raw)
    pred = (m.group(0) == "true") if m else None
    return pred, raw, routed_model


# ──────────────────────────────────────────────────────────────────────────
# 3) Main
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

    for bench in BENCHMARKS:
        print(f"\n=== {bench} ===")
        df_base = load_task_dataframe(bench)
        print(f"Loaded {len(df_base)} rows.")

        for thr in THRESHOLDS:
            run = wandb.init(
                project="routellm-sweep",
                entity="tum-i13",
                name=f"{ROUTER}-{bench}-thr-{thr:.2f}",
                config=dict(benchmark=bench, router=ROUTER, threshold=thr),
                reinit=True,
            )

            # energy-measured inference
            (df, routed_models), energy_j = zeus_energy(
                _infer_dataframe, df_base, client, thr
            )

            acc = (df["predicted"] == df["label"]).mean()
            avg_choice = routed_models.apply(model_choice_to_int).mean()

            wandb.log(dict(accuracy=acc, energy_j=energy_j, model_choice=avg_choice))

            out_dir = results_root / bench
            out_dir.mkdir(exist_ok=True)
            df.to_csv(out_dir / f"{bench}_{thr:.2f}.csv", index=False)

            print(
                f"thr={thr:.2f} | acc={acc:.3%} | "
                f"energy≈{energy_j:.1f} J | model_choice={avg_choice:.2f}"
            )
            run.finish()


def _infer_dataframe(df_base: pd.DataFrame, client, thr):
    df = df_base.copy()
    df[["predicted", "raw_response", "routed_model"]] = df.apply(
        lambda r: pd.Series(get_completion(r, client, thr)), axis=1
    )
    return df, df["routed_model"]


if __name__ == "__main__":
    main()
