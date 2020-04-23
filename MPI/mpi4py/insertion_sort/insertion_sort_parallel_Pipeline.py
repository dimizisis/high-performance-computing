
import numpy as np
from mpi4py import MPI

TAG = 200
ROOT = 0

def fetch_results(sorted_array, n, rank, smaller, comm):
    '''
    Function:  fetch_results 
    --------------------
    Fetches elements from all processes (size in total)

    sorted_array: the array with initial array's sorted elements 
    n: number of elements in the array
    rank: the process that runs the function
    smaller: array's smallest element
    '''
    status = MPI.Status()
    if(rank == ROOT):  # if is process 0
        sorted_array[0] = smaller
        for i in range (1, n):
            sorted_array[i] = comm.recv(source=i, tag=TAG, status=status)
    else:
        comm.send(obj=smaller, dest=ROOT, tag=TAG) # send the smaller element to process 0

def insertion_sort(array, n, rank, comm):
    '''
    Function:  insertion_sort 
    --------------------
    Sorts the array's elements (size in total)
    
    array: the initial array
    n: number of elements in the array
    rank: the process that runs the function
    smaller: array's smallest element
    received: array's element that was last received
    '''
    status = MPI.Status()
    for i in range(n-rank):
        if rank == ROOT:
            received = array[i]
        else:
            received = comm.recv(source=rank-1, tag=TAG, status=status)

        if not i:  # if this is the first step
            smaller = received; # initializes smaller with received at step zero
        else:
            if received > smaller:  # compares smaller to received
                comm.send(obj=received, dest=rank+1, tag=TAG)   # sends the larger number through
            else:
                comm.send(obj=smaller, dest=rank+1, tag=TAG)  # sends the larger number through
                smaller = received
    return smaller

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    a = np.empty(size, dtype=np.int)
    if rank == ROOT:
        lower, upper  = 0, size*4
        a = np.random.randint(low=lower, high=upper, size=size)
        print(f'\nInitial Array: {a}')
        begin = MPI.Wtime()
        
    smaller = insertion_sort(a, size, rank, comm)
    sorted_a = np.empty(size, dtype=np.int)
    fetch_results(sorted_array=sorted_a, n=size, rank=rank, smaller=smaller, comm=comm)

    if rank == ROOT:
        end = MPI.Wtime()
        print(f'\nSorted Array: {sorted_a}')
        print(f'\nTime elapsed for operation: {end-begin} seconds.\n')