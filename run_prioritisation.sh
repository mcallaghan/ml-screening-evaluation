#!/bin/bash                                                                       
#SBATCH --time=10:00:00
#SBATCH --qos=short

#SBATCH --output=out/simulate-ml-prioritisation.out
#SBATCH --error=out/simulate-ml-prioritisation.err
#SBATCH --cpus-per-task=1

module load anaconda/2023.09
module load mpi/intel/5.1.3

source activate ml-screening

export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libmpi.so

srun --mpi=pmi2 -n $1 python screen.py "$2"
