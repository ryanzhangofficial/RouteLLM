import os
import re
import time
import pandas as pd
import wandb

from routellm.controller import Controller

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

##############################################################################
# 1) GLOBALS
##############################################################################
api_key = os.environ.get("HUGGINGFACEHUB_API_TOKEN", "")

# ── Benchmarks ────────────────────────────────────────────────────────────
#   • MMLU block is commented out (kept intact below).
BENCHMARKS = {
    # "MMLU": "routellm/evals/mmlu/responses/",  # dir with many CSVs
    "MT_Bench": "routellm/evals/mt_bench/question.jsonl",
    "GSM8K":    "routellm/evals/gsm8k/gsm8k_responses.csv",
}

# ── Routers to sweep ─────────────────────────────────────────────────────
ROUTER_TYPES = [
    "bert",
    "sw_ranking",
    "mf",
    "causal_llm",
]

# ── Thresholds 0.05 to 0.70 in increments of 0.05 ─────────────────────────
THRESHOLDS = [.05, .20, .45, .70]

##############################################################################
# 2) GPU‑power helper: quick nvidia‑smi snapshot (Watts)
##############################################################################
def get_nvidia_smi_power() -> float:
    """
    Return instantaneous power of GPU‑0 in W.
    If command fails (no GPU/driver) → 0.0
    """
    try:
        cmd = "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits"
        out = os.popen(cmd).read().strip().split("\n")[0]
        return float(out)
    except Exception:
        return 0.0

##############################################################################
# 3) Load benchmark into DF with columns ‘question’, ‘label’
##############################################################################
def load_benchmark_data(benchmark_name: str, path: str) -> pd.DataFrame:
    """
    Return DataFrame with mandatory columns:
        • question  (str)
        • label     (int 0/1)  — 1 means “true”, 0 means “false”.
    """

    # ── MMLU loader (still commented) ───────────────────────────────────
    """
    if benchmark_name == "MMLU":
        import glob
        csv_files = glob.glob(os.path.join(path, "*.csv"))
        if not csv_files:
            raise ValueError(f"No CSVs in {path}")
        dfs = []
        for f in csv_files:
            df_tmp = pd.read_csv(f)
            # Adjust these renames to your actual headers
            rename = {"prompt": "question", "target": "label"}
            df_tmp = df_tmp.rename(columns=rename)
            if "question" not in df_tmp.columns or "label" not in df_tmp.columns:
                raise ValueError(f"{f} missing needed columns.")
            dfs.append(df_tmp)
        return pd.concat(dfs, ignore_index=True)
    """

    # ── MT‑Bench loader ────────────────────────────────────────────────
    if benchmark_name == "MT_Bench":
        df_json = pd.read_json(path, lines=True)
        questions, labels = [], []
        for _, row in df_json.iterrows():
            turns = row.get("turns", [])
            questions.append(turns[0] if turns else "")
            labels.append(1)                # dummy “true”
        return pd.DataFrame({"question": questions, "label": labels})

    # ── GSM8K loader ──────────────────────────────────────────────────
    if benchmark_name == "GSM8K":
        df = pd.read_csv(path)
        # Possible column variants → map to desired names
        rename_map = {
            "prompt": "question",
            "question": "question",
            "input": "question",
            "answer": "label",
            "target": "label",
            "label": "label",
        }
        df = df.rename(columns={c: rename_map[c] for c in df.columns if c in rename_map})

        # If still missing, create dummy label 1 so script won't crash
        if "label" not in df.columns:
            df["label"] = 1
        if "question" not in df.columns:
            raise ValueError("GSM8K file lacks a question column.")

        # Ensure label is int 0/1
        if df["label"].dtype != int:
            df["label"] = (df["label"].astype(str).str.lower() == "true").astype(int)
        return df

    raise ValueError(f"Unknown benchmark {benchmark_name}")

##############################################################################
# 4) One‑row inference helper
##############################################################################
def get_completion(row, client, threshold, router_type):
    prompt = (
        f"Question: {row['question']}\n"
        "Respond with ONLY the word 'true' or 'false':\n"
    )
    messages = [{"role": "user", "content": prompt}]
    model_id = f"router-{router_type}-{threshold}"
    responses, routed_model = client.chat.completions.create(
        model=model_id,
        messages=messages,
    )

    raw = responses[0].outputs[0].text.strip().lower()
    match = re.search(r'\b(true|false)\b', raw)
    pred  = (match.group(0) == "true") if match else None
    return pred, raw, routed_model

##############################################################################
# 5) Main sweep loop
##############################################################################
def main():
    # Login to W&B
    wandb.login()

    for benchmark, path in BENCHMARKS.items():
        print(f"\n=== Benchmark: {benchmark} ===")
        df_base = load_benchmark_data(benchmark, path)
        print(f"Loaded {len(df_base)} rows.")

        for router in ROUTER_TYPES:
            print(f"\n--- Router: {router} ---")
            # Initialize a separate W&B run
            run = wandb.init(
                project="routellm-sweep",
                entity="tum-i13",
                name=f"{benchmark}-{router}",
                reinit=True,
                config={
                    "benchmark": benchmark,
                    "router": router,
                    "thresholds": THRESHOLDS,
                },
            )

            client = Controller(
                routers=[router],
                strong_model="meta-llama/Llama-3.1-8B-Instruct",
                weak_model="meta-llama/Llama-3.2-1B-Instruct",
                api_key=api_key,
                progress_bar=True,
            )

            for thr in THRESHOLDS:
                df = df_base.copy()

                # 5A) GPU power & time start
                p_start = get_nvidia_smi_power()
                t_start = time.time()

                # 5B) Inference
                df[["predicted", "raw_response", "routed_model"]] = df.apply(
                    lambda r: pd.Series(get_completion(r, client, thr, router)),
                    axis=1
                )

                # 5C) Accuracy
                df["correct"] = df.apply(
                    lambda r: r["predicted"] is not None and
                              ((r["predicted"] and r["label"] == 1) or
                               (not r["predicted"] and r["label"] == 0)),
                    axis=1
                )
                acc = df["correct"].mean()

                # 5D) GPU power & time end, approximate energy
                p_end = get_nvidia_smi_power()
                t_end = time.time()
                duration_s = t_end - t_start
                avg_power_w = (p_start + p_end) / 2.0
                energy_j = avg_power_w * duration_s

                # Log to W&B
                wandb.log({
                    "threshold":      thr,
                    "accuracy":       acc,
                    "start_power_w":  p_start,
                    "end_power_w":    p_end,
                    "duration_s":     duration_s,
                    "approx_energy_j": energy_j,
                })

                # Save CSV
                out_dir = f"sweep_results/{benchmark}"
                os.makedirs(out_dir, exist_ok=True)
                out_path = f"{out_dir}/{benchmark}_{router}_{thr:.2f}.csv"
                df.to_csv(out_path, index=False)

                print(
                    f"thr={thr:.2f} | acc={acc:.3%} | "
                    f"energy≈{energy_j:.1f} J | duration={duration_s:.1f}s"
                )

            run.finish()

if __name__ == "__main__":
    main()


