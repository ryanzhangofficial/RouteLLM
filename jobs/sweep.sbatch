#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH --time=1-08:00:00
#SBATCH --array=0-9 
#SBATCH -o output_%A_%a.out
#SBATCH --exclude lrz-hgx-h100-025,lrz-hgx-h100-004,lrz-hgx-h100-030,lrz-hgx-h100-021

# TODO: @Ryan - You need to swap the image (in line 13 of this script) for your own image with your project dependencies.
# TODO: @Ryan - Once you have created that image, you need to reference it on line 27 below.
# TODO: @Ryan - The image that is currently referenced contains all project dependencies that you might need. You can try with the current one first and only if it does not work replace it with your own.
# TODO: @Ryan - Optimally, you specify all paths in absolute format. By that you make sure these don't break.
# TODO: @Ryan - For the container mounts: Only mount them. Don't reassign the paths (see below how I did it).

nvidia-smi
lock=/tmp/messplus.lock
(
  flock -n 200
  enroot create -f --name messplus /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/messplus.sqsh
) 200>"$lock"

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
    /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/ge56heh2/mess-plus/venv/bin/python \
    /dss/dssfs04/lwp-dss-0002/pn72yi/pn72yi-dss-0000/go76xom2/RouteLLM/multi_bench.py \
    $SLURM_ARRAY_TASK_ID     