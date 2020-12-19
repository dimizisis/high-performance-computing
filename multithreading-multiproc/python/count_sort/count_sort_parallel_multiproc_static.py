
import numpy as np
import time
import sys
from multiprocessing import Process, Array

# constants
N = 10
LOWER = 0
UPPER = N*4
###########

def count_sort(a, sorted_a, n, index, rank, blocks):
    stop = rank*blocks + blocks
    for i in range(index, stop):
        count = 0
        for j in range(0, n):
            if a[j] < a[i]:
                count += 1
            elif a[j] == a[i] and j < i:
                count += 1
        sorted_a[count] = a[i]

if __name__ == '__main__':

    try:
        num_proc = int(sys.argv[1])
    except:
        print(f'Usage: {sys.argv[0]} <num_threads>')
        exit(1)

    # initialize array (random)
    array = np.random.randint(low=LOWER, high=UPPER, size=N, dtype=np.int32)

    # create (willing to be) sorted array
    sorted_array = Array('i', N, lock=False)

    # print initial array
    print(array)

    # create sufficient blocks 
    blocks = int (N / num_proc)

    # create processes
    processes = list()
    j = 0
    for i in range(0, num_proc):
        processes.append(Process(target=count_sort, args=(array, sorted_array, N, j, i, blocks,)))
        j += blocks

    # start timer
    begin = time.time()

    # start processes!
    for i in range(0, num_proc):
        processes[i].start()

    # wait all processes!
    for i in range(0, num_proc):
        processes[i].join()

    # stop timer
    end = time.time()

    # print sorted array
    print('[', end=' ')
    [print(f'{sorted_array[j]}', end=' ') for j in range(N)]
    print(']')

    # print time elapsed
    print(f'Time elapsed {end-begin}')
