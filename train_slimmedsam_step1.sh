#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G
#SBATCH --time=75:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-user=xin.peng@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load python/3.10
module load StdEnv/2020
module load  gcc/9.3.0 arrow/9.0.0 opencv/4.8.0
source ../medsam_xp/bin/activate

# pip install -e .
# pip install opencv-python pycocotools matplotlib
python med_prune_distill_step1.py --traindata_path /home/xinpeng/MedSAM/data/npy/CT_Abd_train --valdata_path /home/xinpeng/MedSAM/data/npy/CT_Abd_val --epochs 15 --trainsize 2868 --gradsize 239 --valsize 753
# python prune_distill_step1.py  --traindata_path dataset/small_train --valdata_path dataset/small_val --epochs 1 --trainsize 3 --gradsize 1 --valsize 2
