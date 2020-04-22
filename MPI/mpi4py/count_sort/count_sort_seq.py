
import numpy as np
from mpi4py import MPI

N = 10
LOWER = 0
UPPER = N*4

def count_sort(a, n):
    '''
    Function:  count_sort 
    --------------------
    Sorts an integer array, using the count sort algorithm (enumeration sort)

    a: the array that will be sorted
    n: number of elements in the array=
    '''
    temp = np.empty(dtype=int, shape=n)
    for i in range(n):
        count = 0
        for j in range(n):
            if a[j] < a[i]:
                count += 1
            elif a[j] == a[i] and j < i:
                count += 1
        temp[count] = a[i]

    a[:] = temp

if __name__ == '__main__':
    a = np.random.randint(low=LOWER, high=UPPER, size=N)
    print(f'\nInitial Array: {a}')
    begin = MPI.Wtime()
    count_sort(a, N)
    end = MPI.Wtime()
    print(f'\nSorted Array: {a}')
    print(f'\nTime elapsed for operation: {end-begin} seconds.\n')
