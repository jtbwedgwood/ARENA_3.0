#!/bin/bash
#SBATCH --job-name=arena-test         # Job name (shows up in squeue)
#SBATCH --partition=debug             # Partition to submit to
#SBATCH --qos=debug_qos               # QoS
#SBATCH --gres=gpu:1                  # Request 1 GPU (omit if CPU only)
#SBATCH --cpus-per-task=4             # Number of CPU cores
#SBATCH --mem=16G                     # RAM
#SBATCH --time=01:00:00               # Max runtime (HH:MM:SS)
#SBATCH --output=/home/jwedgwoo/ARENA_3.0/chapter1_transformer_interp/exercises/part2_intro_to_mech_interp/logs/%x_%j.out       # Stdout+stderr (%x=jobname, %j=jobid)
#SBATCH --export=NONE

set -euo pipefail

# --- guard against stray VS Code shell integration anyway ---
unset VSCODE_IPC_HOOK_CLI VSCODE_AGENT_FOLDER VSCODE_NLS_CONFIG \
      VSCODE_HANDLES_UNCAUGHT_ERRORS VSCODE_PID VSCODE_CWD ELECTRON_RUN_AS_NODE
unset -f code 2>/dev/null || true
unset -f code-insiders 2>/dev/null || true
export VSCODE_SHELL_INTEGRATION=false

# Some clusters lack /run/user on compute nodes; avoid it:
export XDG_RUNTIME_DIR="$HOME/.xdg_runtime"
mkdir -p "$XDG_RUNTIME_DIR"; chmod 700 "$XDG_RUNTIME_DIR"

# Always safer to start in your home/project dir
cd $HOME/ARENA_3.0

# Ensure conda is available
source ~/miniconda3/etc/profile.d/conda.sh
conda activate arena-env

# --- run Python unbuffered so logs show immediately ---
export PYTHONUNBUFFERED=1

# Run your Python script
CUDA_LAUNCH_BLOCKING=1 python -u /home/jwedgwoo/ARENA_3.0/chapter1_transformer_interp/exercises/run_block.py /home/jwedgwoo/ARENA_3.0/chapter1_transformer_interp/exercises/part2.py --which latest --setup-cells 1