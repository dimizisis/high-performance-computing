
import numpy as np
from mpi4py import MPI

ROOT = 0
N = 7

def sieve(a, n, locations, start, stop):
    '''
    Function:  sieve 
    --------------------
    Performs sieve

    a: pointer of the array that contains ones
    n: number of elements in the array
    locations: pointer of the array that will contain positions of the non-prime numbers
    start: position from which the process will start
    stop: position that the process will stop
    '''
    for i in range(start, stop):
        j=i
        while (i*j) < n:
            locations[i*j] = 0

def print_primes(a, n):
    '''
    Function:  print_primes 
    --------------------
    Prints prime numbers (position is prime when element of position is 1)

    array: the array that contains 0 and 1 (0 if position is not prime, 1 if it is)
    n: number of elements in the array
    '''
    for i in range (2, n):
        if a[i]:
            print(f'{i} ', end=' ')

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    array = np.empty(dtype=int, shape=N+1)
    if rank == ROOT:
        array = np.ones(shape=N+1)
    
    comm.Bcast(array, ROOT)

    if rank == ROOT:
        begin = MPI.Wtime()

    # These initialization will be done by all processes
    chunk = int(N / size)
    extra = int(N % size)
    start = (rank * chunk)+1
    stop = start + chunk
    if rank == size - 1: stop += extra
    elif rank == ROOT: start += 1

    locations = np.ones(dtype=int, shape=N+1)
    locations[:] = array

    sieve(array, N+1, locations, start, stop)

    comm.Scan(locations, array, MPI.BAND)

    if rank == ROOT:
        end = MPI.Wtime()
        print(f'Primes numbers from 1 to {N} are : ')
        print_primes(array, N+1)
        print(f'\n\nTime spent for operation: {end-begin} seconds')
