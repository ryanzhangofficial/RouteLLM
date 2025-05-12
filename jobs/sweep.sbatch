#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4          # H100 partition
#SBATCH --gres=gpu:1                  # one GPU per task
#SBATCH --array=0-39                  # 40 (benchmark,threshold) pairs, no %-limit
#SBATCH --time=2-00:00:00             # wall-clock upper bound 
#SBATCH -o output_%A_%a.out
#SBATCH --exclude=lrz-hgx-h100-025,lrz-hgx-h100-004,lrz-hgx-h100-030,lrz-hgx-h100-021

set -euo pipefail

image=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/messplus.sqsh
python=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/mess-plus/venv/bin/python
script=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouteLLM/multi_bench.py

thr_values=(0.05 0.20 0.45 0.70)             
bench_idx=$(( SLURM_ARRAY_TASK_ID / 4 ))      
thr_idx=$(( SLURM_ARRAY_TASK_ID % 4 ))       
threshold=${thr_values[$thr_idx]}

container="messplus_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

nvidia-smi  

enroot create -f --name "$container" "$image"

enroot start \
  --root \
  -e HF_TOKEN_PATH=/dss/dsshome1/06/go76xom2/.cache/huggingface/token \
  -e HF_HOME=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/.cache/hf/misc \
  -e HF_DATASETS_CACHE=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/.cache/datasets \
  -e HF_DATASETS_TRUST_REMOTE_CODE=True \
  -e OPENAI_API_KEY=/dss/dsshome1/06/go76xom2/.cache/openai/token \
  -e VLLM_USE_V1=0 \
  --mount /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ \
  --mount /dss/dsshome1/06/go76xom2/ \
  "$container" \
  "$python" "$script" "$bench_idx" "$threshold"
