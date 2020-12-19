
import sys
from multiprocessing import Process, Value
import time
import random

def pi(count):
    x = random.random()
    y = random.random()
    z = x*x+y*y
    if z <= 1:
        with count.get_lock(): # recursive lock object (because we created count with lock=True)
            count.value += 1

if __name__ == '__main__':

    try:
        niter = int(sys.argv[1])    # num points
    except:
        print(f'Usage: {sys.argv[0]} <num_points>')
        exit(1)

    # create shared value (int), need protection by lock
    count = Value('i', lock=True)
    
    # create processes
    processes = list()
    for i in range(0, niter):
        processes.append(Process(target=pi, args=(count,)))

    # start timer
    begin = time.time()

    # start processes!
    for i in range(0, niter):
        processes[i].start()

    # wait all processes!
    for i in range(0, niter):
        processes[i].join()

    pi = count.value/niter*4

    # stop timer
    end = time.time()

    # print pi
    print(pi)

    # print time elapsed
    print(f'Time elapsed {end-begin}')