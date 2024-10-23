#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu1
##
#SBATCH --job-name=simulation0_1
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err

hostname
date


module add R/4.3.1

export R_LIBS="/home1/mose1103/R/library"

for argv in "$*"
do
    $argv
done
