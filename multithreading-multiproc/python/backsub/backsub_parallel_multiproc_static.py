
import numpy as np
import time
import sys
from multiprocessing import Process, Array

def backsub(a, b, x, rank, idx, blocks):
    stop = rank*blocks + blocks
    for i in range(idx, stop):
        s = 0.0
        for j in range(i):
            s = s + (x[j] * a[i][j])
        x[i] = (b[i] - s) / a[i][i]

if __name__ == '__main__':
    try:
        N = int(sys.argv[1])
        num_proc = int(sys.argv[2])
    except:
        print(f'Usage: {sys.argv[0]} <size> <num_threads>')
        exit(1)

    # initialize array (random)
    a = np.random.rand(N,N)

    b = np.empty(shape=N, dtype=np.float)

    # create (willing to be) sorted array
    x_shared = Array('d', N, lock=False)

    # print A & B
    print(a)
    print(b)

    # create sufficient blocks 
    blocks = int (N / num_proc)

    # create processes
    processes = list()
    j = 0
    for i in range(0, num_proc):
        processes.append(Process(target=backsub, args=(a, b, x_shared, i, j, blocks,)))
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
    [print(f'{x_shared[j]}', end=' ') for j in range(N)]
    print(']')

    # print time elapsed
    print(f'Time elapsed {end-begin}')
