
import sys
from threading import Thread
import time
import random

def pi(count):
    x = random.random()
    y = random.random()
    z = x*x+y*y
    if z <= 1:
        count[0] += 1 # built-in structure, thread safe

if __name__ == '__main__':

    try:
        niter = int(sys.argv[1])    # num points
    except:
        print(f'Usage: {sys.argv[0]} <num_points>')
        exit(1)

    # create list to pass it as reference
    count = [0]
    
    # create threads
    threads = list()
    for i in range(0, niter):
        threads.append(Thread(target=pi, args=(count,)))

    # start timer
    begin = time.time()

    # start threads!
    for i in range(0, niter):
        threads[i].start()

    # wait all threads!
    for i in range(0, niter):
        threads[i].join()

    pi = count[0]/niter*4

    # stop timer
    end = time.time()

    # print pi
    print(pi)

    # print time elapsed
    print(f'Time elapsed {end-begin}')