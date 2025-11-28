#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:rtx:1
#SBATCH --mem-per-gpu 50G
#SBATCH --output slurm-jupyter.out
#SBATCH --dependency singleton
#SBATCH --job-name jupyterlab

# Get the current user's UID
current_uid=$(id -u)

# Perform the calculation: (UID - 1000000 + 38880)
port=$((current_uid - 1000000 + 38880))

singularity exec -B /scratch --env PYTHONPATH=`pwd`:`pwd`/calo_cluster  --nv /home/software/singularity/mlpf-torchsparse.simg jupyter lab --no-browser --ip=0.0.0.0 --notebook-dir=$HOME --port $port
