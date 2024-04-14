#!/bin/bash
#SBATCH --gpus-per-node=a100:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --account=def-annielee
#SBATCH --mail-user=xin.peng@mail.utoronto.ca
#SBATCH --mail-type=ALL

module load python/3.10
module load StdEnv/2020
module load  gcc/9.3.0 arrow/9.0.0 opencv/4.8.0
source ../medsam_xp/bin/activate

python prune_distill_step2.py --model_path checkpoints/vit_b_medslim_final_step1_50.pth --traindata_path /home/xinpeng/MedSAM/data/npy/CT_Abd_train --valdata_path /home/xinpeng/MedSAM/data/npy/CT_Abd_val --epochs 15 --trainsize 2868 --gradsize 239 --valsize 753
