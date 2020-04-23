
import numpy as np
from mpi4py import MPI

N = 10
LOWER = 0
UPPER = N*4

def insertion_sort(a, n):
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
    for i in range(1, n):
        j = i-1 
        key = a[i]
        while (a[j] > key) and (j >= 0):
           a[j+1] = a[j]
           j -= 1
        a[j+1] = key

if __name__ == '__main__':
    a = np.random.randint(low=LOWER, high=UPPER, size=N)
    print(f'\nInitial Array: {a}')
    begin = MPI.Wtime()
    insertion_sort(a, N)
    end = MPI.Wtime()
    print(f'\nSorted Array: {a}')
    print(f'\nTime elapsed for operation: {end-begin} seconds.\n')