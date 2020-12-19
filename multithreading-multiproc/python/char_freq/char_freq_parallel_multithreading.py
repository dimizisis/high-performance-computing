
import sys
from os import SEEK_END
import time
import numpy as np
from threading import Thread, RLock

NUM_OF_ARGS = 3
N = 128
base = 0

def check_args():
    '''
    Function:  check_args 
    --------------------
    Checks if the number of arguments is correct
    '''
    # if no filename given, exit
    if len(sys.argv) != NUM_OF_ARGS:
        print(f'Usage : {sys.argv[0]} <file_name> <num_threads>')
        exit(1)
    return sys.argv[1], int(sys.argv[2])

def open_file(mode='r'):
    '''
    Function:  open_file 
    --------------------
    Opens input file size
    mode: method (mode) for opening the file
    '''
    try:
        f = open(filename, mode)
    except Exception as e:
        print(e)
        exit(2)
    return f

def obtain_file_size(f):
    '''
    Function:  obtain_file_size 
    --------------------
    Obtains input file's size
    f: input file
    '''
    f.seek(0, SEEK_END)
    file_size = f.tell()
    rewind(f)
    return file_size

def count_characters(freq, buffer, tid, idx, blocks, lock):
    '''
    Function:  count_characters 
    --------------------
    Counts the frequency of characters in an char array
    
    freq: the array that will contain each character's frequency
    buffer: the array that contains the characters
    tid: thread's id
    idx: start index of thread
    blocks: number of blocks
    lock: shared lock among threads
    '''
    stop = tid * blocks + blocks
    for i in range(idx, stop):
        lock.acquire()  # no need for locks, because of the GIL (for multithreading only)
        freq[buffer[i] - base] += 1
        lock.release()

def rewind(f):
    '''
    Function:  rewind 
    --------------------
    Set the file position to the beginning of the file 
    of the given stream

    f: input file
    '''
    f.seek(0)

if __name__ == '__main__':

    # check arguments, if file name & number of threads are given, return them
    filename, num_threads = check_args()

    # open file (default mode='r')
    f = open_file()

    # get file size
    file_size = obtain_file_size(f)
    
    # print file size
    print(f'File size is {file_size}\n')

    # frequency vector init with zeros
    freq = np.zeros(N, dtype=np.int32)

    # convert buffer to ascii buffer (numpy array)
    ascii_buff = np.asarray([ord(char) for char in list(f.read(file_size))], dtype=np.int32)

    # create sufficient blocks 
    blocks = int(file_size /num_threads)

    # create shared lock
    lock = RLock()

    # create threads
    threads = list()
    j = 0
    for i in range(num_threads):
        threads.append(Thread(target=count_characters, args=(freq, ascii_buff, i, j, blocks, lock,)))
        j += blocks

    # start counting
    start = time.time()

    # start threads
    for i in range(num_threads):
        threads[i].start()

    # wait all threads to finish
    for i in range(num_threads):
        threads[i].join()

    # done
    end = time.time()

    # print char frequency
    [print(f'{j+base} = {freq[j]}') for j in range(N)]

    # print time of execution
    print(f'Time spent for counting: {end-start}')

    # close file
    f.close()
