#!/bin/bash
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=1
#SBATCH --partition=cpu
#SBATCH --out=/home/rektomar/logs/hmill_mip/%x-%j.out

# n - ix of config in grid, m - ix of dataset
#=
ml --ignore_cache Julia/1.8.0-linux-x86_64 
srun julia scripts/hmillclassifier.jl --n $1 --m $2
exit
# =#


