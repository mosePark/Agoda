#!/bin/bash

# virutal environment directory
ENV=/home1/mose1103/anaconda3/envs/agoda_R/bin/Rscript

# file directory of multiple execution source ".sh"
RUN_SRC=/home1/mose1103/agoda/simulation/code/run_src_01.sh

# file directory of experiment ".R"
EXECUTION_FILE=/home1/mose1103/agoda/simulation/code/temp01.R

# default prefix of job name
DEFAULT_NAME=agoda

# R argparse source for experiments

sz=(
"--sz 3000 --realization 20"
"--sz 5000 --realization 12"
"--sz 8000 --realization 8"
"--sz 32083 --realization 5"
)

# realization=(
# "--realization 20"
# "--realization 12"
# "--realization 8"
# "--realization 5"
# )


k=(
"--k 3"
"--k 4"
"--k 5"
)

clrepN=(
"--clrepN 10"
)

dim=(
"--dim 15"
)


for i1 in ${!sz[*]}; do
    # Pair sz and realization
    for i2 in ${!k[*]}; do
        for i3 in ${!clrepN[*]}; do
            sbatch --job-name=$DEFAULT_NAME $RUN_SRC $ENV $EXECUTION_FILE ${sz[$i1]} ${k[$i2]} ${clrepN[$i3]} ${dim[0]}
            # echo --job-name=$DEFAULT_NAME $RUN_SRC $ENV $EXECUTION_FILE ${sz[$i1]} ${k[$i2]} ${clrepN[$i3]} --dim 15
            sleep 1
        done
    done
done
