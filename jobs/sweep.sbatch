#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=1-08:00:00
#SBATCH --array=0-9               # 10 tasks
#SBATCH -o output_%A_%a.out
#SBATCH --exclude=lrz-hgx-h100-025,lrz-hgx-h100-004,lrz-hgx-h100-030,lrz-hgx-h100-021

set -euo pipefail

# absolute paths
image=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/messplus.sqsh
python=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/mess-plus/venv/bin/python
script=/dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouteLLM/multi_bench.py

# make this unique per array task:
container="messplus_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

nvidia-smi

# force-create brand-new container in its own directory
enroot create -f --name "$container" "$image"

# start it
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
    "$python" "$script" "$SLURM_ARRAY_TASK_ID"
