#!/bin/bash
#SBATCH --job-name=hr
#SBATCH --partition=normal
#SBATCH -N 1
#SBATCH --time=120:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mail-type=none
#SBATCH --mail-user=zhaojh1@shanghaitech.edu.cn
#SBATCH --output=hr.out
#SBATCH --error=hr.err
python tools/train.py --cfg experiments/hrnet_w18.yaml
