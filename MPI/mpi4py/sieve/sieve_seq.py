
import numpy as np
from mpi4py import MPI

N = 12

def sieve(a, n):
    '''
    Function:  sieve 
    --------------------
    Performs sieve

    a: pointer of the array that contains ones
    n: number of elements in the array
    '''
    for i in range (2, n+1):
        if a[i]:
            j=i
            while (i*j) <= n:
                a[(i*j)] = 0
                j+=1

def print_primes(array, n):
    '''
    Function:  print_primes 
    --------------------
    Prints prime numbers (position is prime when element of position is 1)

    array: the array that contains 0 and 1 (0 if position is not prime, 1 if it is)
    n: number of elements in the array
    '''
    for i in range (2, n):
        if array[i]:
            print(f'{i} ', end=' ')

if __name__ == '__main__':
    array = np.ones(dtype=int, shape=N+1)
    begin = MPI.Wtime()
    sieve(array,N)
    end = MPI.Wtime()
    print(f'Primes numbers from 1 to {N} are : ')
    print_primes(array, N+1)
    print(f'\n\nTime spent for operation: {end-begin} seconds')