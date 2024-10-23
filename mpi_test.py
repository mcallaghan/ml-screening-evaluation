from mpi4py import MPI
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
print(rank)

#f = h5py.File('p/tmp/maxcall/ml-screening/parallel_test.hdf5','w',driver='mpio',comm=MPI.COMM_WORLD)

#dset = f.create_dataset('test', (4,), dtype='i')
#dset[rank] = rank

size = 10000
x = np.zeros(size)
x[:] = rank
y = np.arange(size)
table = pa.table({'rank': x, 'y': y})
pq.write_to_dataset(table,root_path='/p/tmp/maxcall/ml-screening/ptest',partition_cols=['rank'])