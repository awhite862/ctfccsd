import time
from pyscf import lib
from pyscf.lib import logger

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

INQUIRY = 50040
class omp(lib.with_omp_threads):
    def __init__(self, nthreads=None, interval=0.05):
        self.interval = interval
        if nthreads is None:
            nthreads = size
        if rank == 0:
            lib.with_omp_threads.__init__(self, nthreads)

    def __enter__(self):
        if rank == 0:
            lib.with_omp_threads.__enter__(self)

    def __exit__(self, type, value, traceback):
        if rank == 0:
            lib.with_omp_threads.__exit__(self, type, value, traceback)
            for i in range(1,size):
                comm.send('Done', i, tag=INQUIRY)
        else:
            while True:
                time.sleep(self.interval)
                if comm.Iprobe(source=0, tag=INQUIRY):
                    task = comm.recv(source=0, tag=INQUIRY)
                    if task == 'Done':
                        break


class Logger(logger.Logger):
    def __init__(self, stdout, verbose):
        if rank == 0:
            logger.Logger.__init__(self, stdout, verbose)
        else:
            logger.Logger.__init__(self, stdout, 0)


def static_partition(tasks):
    segsize = (len(tasks)+size-1) // size
    start = rank * segsize
    stop = min(len(tasks), start+segsize)
    return tasks[start:stop]
