#!/bin/bash
#SBATCH --job-name=dim_2
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err
#SBATCH --nodes=001
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu1

hostname
date

module add R/4.3.1

export R_LIBS="/home1/mose1103/R/library"

/home1/mose1103/anaconda3/envs/agoda_R/bin/Rscript dim2.R
