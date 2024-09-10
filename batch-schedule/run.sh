'''
sbatch baseline
'''

#!/bin/bash
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --partition=gpu
##
#SBATCH --job-name=dt
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err
##
#SBATCH --gres=gpu:rtx3090:1

hostname
date

module add CUDA/11.3.0
module add ANACONDA/2020.11




python3 run_seq_cls.py --task nsmc --config_file KcELECTRA-v2022-0404-800k-discriminator.json
python3 run_seq_cls.py --task kornli --config_file KcELECTRA-v2022-0404-800k-discriminator.json
python3 run_seq_cls.py --task paws --config_file KcELECTRA-v2022-0404-800k-discriminator.json
python3 run_seq_cls.py --task question-pair --config_file KcELECTRA-v2022-0404-800k-discriminator.json
python3 run_seq_cls.py --task korsts --config_file KcELECTRA-v2022-0404-800k-discriminator.json
python3 run_ner.py --task naver-ner --config_file KcELECTRA-v2022-0404-800k-discriminator.json
python3 run_squad.py --task korquad --config_file KcELECTRA-v2022-0404-800k-discriminator.json



'''
AL 때 예시
'''



#!/bin/bash
#SBATCH --job-name=AL
#SBATCH -o SLURM.%N.%j.out
#SBATCH -e SLURM.%N.%j.err
#SBATCH --nodes=001
#SBATCH --ntasks-per-node=1
#SBATCH --partition=gpu3

hostname
date

module add ANACONDA/2020.11

/home1/mose1103/anaconda3/envs/fine-koactive/bin/python main.py --task active_learning
