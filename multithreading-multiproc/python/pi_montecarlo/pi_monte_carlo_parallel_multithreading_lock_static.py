
import sys
from threading import Thread, RLock
import time
import random

class Counter:

    # these are CLASS variables, not instance variables :-)
    # class variables -> static (in Java)
    lock = RLock()
    count = 0
    #######################################################

    def increment(self):
        self.lock.acquire()
        self.count += 1
        self.lock.release()

def calc_count(count, rank, idx, blocks):
    stop = rank * blocks + blocks
    for i in range(idx, stop):
        x = random.random()
        y = random.random()
        z = x*x+y*y
        if z <= 1:
            count.increment()

if __name__ == '__main__':

    try:
        niter = int(sys.argv[1])    # num points
        num_threads = int(sys.argv[2])
    except:
        print(f'Usage: {sys.argv[0]} <num_points> <num_threads>')
        exit(1)

    # create list to pass it as reference
    counter = Counter()

    # create sufficient blocks
    blocks = int(niter / num_threads)
    
    # create threads
    threads = list()
    j = 0
    for i in range(0, niter):
        threads.append(Thread(target=calc_count, args=(counter, i, j, blocks)))
        j += blocks

    # start timer
    begin = time.time()

    # start threads!
    for i in range(0, num_threads):
        threads[i].start()

    # wait all threads!
    for i in range(0, num_threads):
        threads[i].join()

    pi = counter.count/niter*4

    # stop timer
    end = time.time()

    # print pi
    print(pi)

    # print time elapsed
    print(f'Time elapsed {end-begin}')