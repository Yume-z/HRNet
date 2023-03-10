#!/bin/bash
#SBATCH --job-name=hr
#SBATCH --partition=debug
#SBATCH -N 1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:2
#SBATCH --mail-type=none
#SBATCH --mail-user=zhaojh1@shanghaitech.edu.cn
#SBATCH --output=hr.out
#SBATCH --error=hr.err
python tools/train.py --cfg experiments/hrnet_us.yaml
