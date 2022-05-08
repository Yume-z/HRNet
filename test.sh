#!/bin/bash
#SBATCH --job-name=hr
#SBATCH --partition=debug
#SBATCH -N 1
#SBATCH --gres=gpu:2
#SBATCH --mail-type=none
#SBATCH --mail-user=zhaojh1@shanghaitech.edu.cn
#SBATCH --output=hr.out
#SBATCH --error=hr.err
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/0best_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/1best_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/2best_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/3best_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/4best_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/5best_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/6best_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/7best_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/8best_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/9best_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/0final_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/1final_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/2final_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/3final_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/4final_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/5final_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/6final_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/7final_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/8final_state.pth
python tools/test.py --cfg experiments/hrnet_w18.yaml --model-file ./output/Xray/hrnet_w18/9final_state.pth