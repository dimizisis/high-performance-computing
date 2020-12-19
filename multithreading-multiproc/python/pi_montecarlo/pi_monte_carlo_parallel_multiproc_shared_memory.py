
import sys
from multiprocessing import Process, Pool
from multiprocessing.managers import SharedMemoryManager
import time
import random

class Count:
    def __init__(self):
        self.val = 0
    def inc(self):
        self.val += 1

def calc_count(count):
    # for i in range(niter):
    x = random.random()
    y = random.random()
    z = x*x+y*y
    if z <= 1:
        count.inc()

if __name__ == '__main__':

    try:
        niter = int(sys.argv[1])    # num points
        num_proc = int(sys.argv[2])
    except:
        print(f'Usage: {sys.argv[0]} <num_points> <num_proc>')
        exit(1)

    chunk = int(niter/num_proc)

    # start timer
    begin = time.time()

    # start num_proc worker processes
    with Pool(processes=num_proc) as pool:
        count = Count()
        res = pool.starmap(calc_count, chunksize=chunk, iterable=[(count, )])
        print(res)

    # stop timer
    end = time.time()

    # print time elapsed
    print(f'Time elapsed {end-begin}')