#!/bin/bash
#SBATCH --qos=gpumedium                                                                           
#SBATCH --time=00:10:00

#SBATCH --output=out/setup.out
#SBATCH --error=out/setup.err
#SBATCH --ntasks=1
#SBATCH --partition=gpu                      
#SBATCH --gres=gpu:v100:1

export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libpmi.so

module load anaconda/2023.09
conda create --name llm-screening python=3.11
source activate llm-screening
pip install -r llm_requirements.txt
