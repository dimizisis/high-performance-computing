
import sys
from multiprocessing import Process, Value
import time
import random

def calc_count(count, rank, idx, blocks):
    stop = rank * blocks + blocks
    for i in range(idx, stop):
        x = random.random()
        y = random.random()
        z = x*x+y*y
        if z <= 1:
            with count.get_lock(): # recursive lock object (because we created count with lock=True)
                count.value += 1

if __name__ == '__main__':

    try:
        niter = int(sys.argv[1])    # num points
        num_proc = int(sys.argv[2])
    except:
        print(f'Usage: {sys.argv[0]} <num_points> <num_proc>')
        exit(1)

    # shared value among processes (needs lock protection in mp!)
    count = Value('i', lock=True)

    # create sufficient blocks
    blocks = int(niter / num_proc)
    
    # create processes
    processes = list()
    j = 0
    for i in range(0, num_proc):
        processes.append(Process(target=calc_count, args=(count, i, j, blocks,)))
        j += blocks

    # start timer
    begin = time.time()

    # start processes!
    for i in range(0, num_proc):
        processes[i].start()

    # wait all processes!
    for i in range(0, num_proc):
        processes[i].join()

    pi = count.value/niter*4

    # stop timer
    end = time.time()

    # print pi
    print(pi)

    # print time elapsed
    print(f'Time elapsed {end-begin}')