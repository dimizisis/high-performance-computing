
import numpy as np
from mpi4py import MPI

N = 12
LOWER = 0
UPPER = N*4
ROOT = 0

def count_sort(a, n, start, stop):
    '''
    Function:  count_sort 
    --------------------
    Sorts an integer array, using the count sort algorithm (enumeration sort)

    a: the array that will be sorted
    n: number of elements in the array
    '''
    temp = np.zeros(dtype=int, shape=a.size)
    for i in range(start, stop):
        count = 0
        for j in range(n):
            if a[j] < a[i]:
                count += 1
            elif a[j] == a[i] and j < i:
                count += 1
        temp[count] = a[i]

    a[:] = temp

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    a = np.empty(dtype=int, shape=N)
    sorted_a = np.empty(dtype=int, shape=N)
    if rank == ROOT:
        a = np.random.randint(low=LOWER, high=UPPER, size=N)
        print(f'Initial Array: {a}')

    comm.Bcast(a, ROOT)
    
    if rank == ROOT:
        # if rank is 0, start counting
        begin = MPI.Wtime()

    # These initialization will be done by all processes
    chunk = int(N / size)
    extra = int(N % size)
    start = rank * chunk
    stop = start + chunk
    if rank == size - 1: stop += extra

    count_sort(a, N, start, stop)

    comm.Reduce(a, sorted_a, MPI.SUM, ROOT)

    if rank == ROOT:
        end = MPI.Wtime()
        print(f'Sorted Array: {sorted_a}')
        print(f'Time elapsed for operation: {end-begin} seconds.')
