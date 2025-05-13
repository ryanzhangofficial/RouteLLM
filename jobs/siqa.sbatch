#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH -o output_%A.out
#SBATCH --exclude lrz-hgx-h100-025

cd "$(dirname "$0")"

nvidia-smi
enroot create /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/messplus.sqsh

# Train
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
    --mount /sbin \
    --mount /usr/share/ \
    messplus \
    /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/mess-plus/venv/bin/python /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouteLLM/routellm_bench.py \
    --config /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouteLLM/config/pretrained/siqa.yaml \
    --wandb-entity tum-i13 \
    --project-name routellm-sweepv2
