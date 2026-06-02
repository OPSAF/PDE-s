#!/bin/bash

# 激活conda环境
source /mnt/c/Users/27862/anaconda3/etc/profile.d/conda.sh
conda activate /mnt/c/Users/27862/anaconda3/envs/P

# 运行PINN脚本
python ex_pinn.py "$@"