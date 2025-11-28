#!/bin/bash
#SBATCH --partition gpu
#SBATCH --gres gpu:l40:1
#SBATCH --output slurm-jupyter.out
#SBATCH --job-name mlpf-cluster
#SBATCH --ntasks=1                     # number of tasks (usually 1 per job)
#SBATCH --cpus-per-task=12             # number of CPUs per task
#SBATCH --mem=90000

singularity exec -B /scratch --env PYTHONPATH=`pwd`:`pwd`/calo-cluster/calo_cluster --nv /home/software/singularity/mlpf-torchsparse.simg python3 train.py
