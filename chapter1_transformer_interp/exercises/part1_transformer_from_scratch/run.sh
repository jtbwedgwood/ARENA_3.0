#!/bin/bash
#SBATCH --job-name=arena-test         # Job name (shows up in squeue)
#SBATCH --partition=debug             # Partition to submit to
#SBATCH --qos=debug_qos               # QoS
#SBATCH --gres=gpu:1                  # Request 1 GPU (omit if CPU only)
#SBATCH --cpus-per-task=4             # Number of CPU cores
#SBATCH --mem=16G                     # RAM
#SBATCH --time=01:00:00               # Max runtime (HH:MM:SS)
#SBATCH --output=/home/jwedgwoo/ARENA_3.0/chapter1_transformer_interp/exercises/part1_transformer_from_scratch/logs/%x_%j.out       # Stdout+stderr (%x=jobname, %j=jobid)

# Always safer to start in your home/project dir
cd $HOME/ARENA_3.0

# Ensure conda is available
source ~/miniconda3/etc/profile.d/conda.sh
conda activate arena-env

# Run your Python script
CUDA_LAUNCH_BLOCKING=1 python /home/jwedgwoo/ARENA_3.0/chapter1_transformer_interp/exercises/run_block.py /home/jwedgwoo/ARENA_3.0/chapter1_transformer_interp/exercises/part1.py --which latest --setup-cells 1