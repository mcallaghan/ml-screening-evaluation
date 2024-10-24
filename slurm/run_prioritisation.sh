#!/bin/bash                                                                       
#SBATCH --time=23:59:00
#SBATCH --qos=short

#SBATCH --output=out/simulate-ml-prioritisation.out
#SBATCH --error=out/simulate-ml-prioritisation.err
#SBATCH --cpus-per-task=1

module load python/3.12.3
module load intel/oneAPI/2024.0.0

source cpuvenv/bin/activate

#export I_MPI_PMI_LIBRARY=/p/system/slurm/lib/libmpi.so


#rm -rf /p/tmp/maxcall/ml-screening/ordered_records
#rm -rf /p/tmp/maxcall/ml-screening/batch_predictions

python setup_db.py
#python setup_db.py --delete 1

mpirun -np $1 python screen.py "$2"
######srun --mpi=pmi2 -n $1 python screen.py "$2"
