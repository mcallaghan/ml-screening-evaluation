#!/bin/bash                                                                       
#SBATCH --time=23:59:00
#SBATCH --qos=short

#SBATCH --output=out/stopping-crit.out
#SBATCH --error=out/stopping-crit.err
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=100

module load python/3.12.3
module load intel/oneAPI/2024.0.0

source cpuvenv/bin/activate

mpirun -np 100 python stopping.py 0.0

