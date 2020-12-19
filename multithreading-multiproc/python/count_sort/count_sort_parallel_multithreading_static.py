
import numpy as np
import time
import sys
from threading import Thread

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
        num_threads = int(sys.argv[1])
    except:
        print(f'Usage: {sys.argv[0]} <num_threads>')
        exit(1)

    # initialize array (random)
    array = np.random.randint(low=LOWER, high=UPPER, size=N, dtype=np.int32)

    # create (willing to be) sorted array
    sorted_array = np.empty(shape=N, dtype=np.int32)

    # print initial array
    print(array)

    # create sufficient blocks 
    blocks = int (N / num_threads)

    # create threads
    threads = list()
    j = 0
    for i in range(0, num_threads):
        threads.append(Thread(target=count_sort, args=(array, sorted_array, N, j, i, blocks,)))
        j += blocks

    # start timer
    begin = time.time()

    # start threads!
    for i in range(0, num_threads):
        threads[i].start()

    # wait all threads!
    for i in range(0, num_threads):
        threads[i].join()

    # stop timer
    end = time.time()

    # print sorted array
    print(sorted_array)

    # print time elapsed
    print(f'Time elapsed {end-begin}')
