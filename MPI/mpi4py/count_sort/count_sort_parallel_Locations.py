
import numpy as np
from mpi4py import MPI

N = 8
LOWER = 0
UPPER = N
ROOT = 0

def count_sort(a, n, start, stop, locations):
    '''
    Function:  count_sort 
    --------------------
    Sorts an integer array, using the count sort algorithm (enumeration sort)

    a: the array that will be sorted
    n: number of elements in the array
    start: start index of process p
    stop: stop indes of process p
    '''
    for i in range(start, stop):
        count = 0
        for j in range(n):
            if a[j] < a[i]:
                count += 1
            elif a[j] == a[i] and j < i:
                count += 1
        locations[i - start] = count

def attach_results(buff, n, results, locations):
    '''
    Function:  attatch_results 
    --------------------
    Attaches final results to results array, using a locations vector
    
    buff: the initial array
    results: the results array, which will contain the elements of buff sorted
    locations: locations vector, contains the right positions for buff's elements (0...N, in order the elements to be sorted)
    '''
    for i in range(n):
        results[locations[i]] = buff[i]

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    locations = None
    sorted_a = None
    a = np.empty(dtype=int, shape=N)
    if rank == ROOT:
        a = np.random.randint(low=LOWER, high=UPPER, size=N)
        print(f'Initial Array: {a}')

    comm.Bcast(a, ROOT)
    
    if rank == ROOT:
        # if rank is 0, start counting
        begin = MPI.Wtime()

    # These initialization will be done by all processes
    chunk = int(N / size)
    start = rank * chunk
    stop = start + chunk

    local_locations = np.empty(dtype=int, shape=(stop-start))

    count_sort(a, N, start, stop, local_locations)

    if rank == ROOT:
        locations = np.empty(dtype=int, shape=N)
        comm.Gather(local_locations, locations, ROOT)
        sorted_a = np.empty(dtype=int, shape=N)
        attach_results(a, N, sorted_a, locations)
        end = MPI.Wtime()
        print(f'Sorted Array:  {sorted_a}')
        print(f'Time elapsed for operation: {end-begin} seconds.')
    else:
        comm.Gather(local_locations, locations, ROOT)
