#!/bin/bash
#SBATCH --job-name=simul0_1
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err
#SBATCH --nodes=004
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu1

hostname
date


module add R/4.3.1

for argv in "$*"
do
    $argv
done
