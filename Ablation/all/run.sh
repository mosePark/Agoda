#!/bin/bash
#SBATCH --job-name=temp1.5-gen1
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err
#SBATCH --nodes=002
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu1

hostname
date

module add ANACONDA/2020.11

/home1/mose1103/anaconda3/envs/agoda/bin/python temp1.5-gen1.py
/home1/mose1103/anaconda3/envs/agoda/bin/python temp1.5-gen2.py
/home1/mose1103/anaconda3/envs/agoda/bin/python temp0.1-gen1.py
/home1/mose1103/anaconda3/envs/agoda/bin/python temp0.1-gen2.py
