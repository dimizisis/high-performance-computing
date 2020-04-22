
import numpy as np
from mpi4py import MPI

N = 10
LOWER = 0
UPPER = N*4

def shell_sort(array, n):
    '''
    Function:  shell_sort 
    --------------------
    Sorts an integer array, using the shell sort algorithm

    a: the array that will be sorted
    n: number of elements in the array
    '''
    h = int(n/2)
    while h > 0:
        for i in range (h, n):
            t = array[i]
            j=i
            while j >= h and t < array[j-h]:
                array[j] = array[j - h]
                j -= h
            array[j] = t
        h = int(h/2)

if __name__ == '__main__':
    a = np.random.randint(low=LOWER, high=UPPER, size=N)
    print(f'\nInitial Array: {a}')
    begin = MPI.Wtime()
    shell_sort(a, N)
    end = MPI.Wtime()
    print(f'\nSorted Array: {a}')
    print(f'\nTime elapsed for operation: {end-begin} seconds.\n')