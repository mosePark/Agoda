#!/bin/bash
#SBATCH --nodes=1
#SBATCH --partition=gpu2
#SBATCH --gres=gpu:a10:1
#SBATCH --cpus-per-task=2
#SBATCH -o ./_out/%j.sbatch.%N.out
#SBATCH -e ./_err/%j.sbatch.%N.err


#=============================================================
GRES="gpu:a10:1"                   
. gosdt-conf.sh
#==============================================================

# 컨테이너 생성 및 시작
enroot create --name $CONTAINER_NAME $CONTAINER_IMAGE_PATH
enroot start --rw $CONTAINER_NAME -- bash -c "
    apt-get update && \
    apt-get install -y cmake ninja-build pkg-config patchelf libtbb-dev libgmp-dev && \
    pip3 install --upgrade scikit-build auditwheel
"
