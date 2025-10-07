#!/bin/bash
#SBATCH --job-name=pointtransformer_test
#SBATCH --output="/work/vezzani_fakegs/GaussianDiscriminator/logs/traingauss_%A.out"
##SBATCH --array=0
#SBATCH --partition=all_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --account=vezzani_fakegs
#SBATCH --time=1-00:00:00
#SBATCH --mem=48G
##SBATCH --dependency=afterany:2489554
#SBATCH --cpus-per-gpu=8
#SBATCH --constraint="gpu_A40_48G|gpu_L40S_48G|gpu_RTX6000_24G|gpu_RTXA5000_24G"


module unload cuda/12.6
module load cuda/11.8

source activate /homes/rcatalini/.conda/envs/fake_gaussian

python -u train_point.py

