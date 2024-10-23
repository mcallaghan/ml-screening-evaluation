module load python/3.12.3
module load intel/oneAPI/2024.0.0
#module load openmpi/5.0.3
#module load hdf5/1.14.4

source cpuvenv/bin/activate

mpirun -np 100 python mpi_test.py