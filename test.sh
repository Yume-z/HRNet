#!/bin/bash
#SBATCH --job-name=hr
#SBATCH --partition=debug
#SBATCH -N 1
#SBATCH --gres=gpu:1
#SBATCH --mail-type=none
#SBATCH --mail-user=zhaojh1@shanghaitech.edu.cn
#SBATCH --output=hr.out
#SBATCH --error=hr.err
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file 0best_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file 1best_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file 2best_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file 3best_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file 4best_state.pth