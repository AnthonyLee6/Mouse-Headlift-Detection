#!/bin/bash
set -e
source /root/anaconda3/etc/profile.d/conda.sh
conda activate dlc_tmp
export CUDA_VISIBLE_DEVICES=0
export TF_FORCE_GPU_ALLOW_GROWTH=true
cd /root/dlc_infer_2166_2171
nohup bash -c 'source /root/anaconda3/etc/profile.d/conda.sh && conda activate dlc_tmp && export CUDA_VISIBLE_DEVICES=0 && python run_infer.py' > infer.log 2>&1 &
echo "INFER_PID $!"
sleep 2
tail -20 infer.log || true
