#!/bin/bash
set -e
source /root/anaconda3/etc/profile.d/conda.sh
if conda env list | grep -q dlc_tmp; then
  echo ENV_EXISTS
else
  conda create -n dlc_tmp python=3.9 -y
fi
conda activate dlc_tmp
python -V
pip install -U pip -q
pip install "deeplabcut[tf]"
python -c "import deeplabcut as d; print('DLC_OK', d.__version__)"
