
import sys
from os import SEEK_END
import time
import numpy as np
from multiprocessing import Process, Lock, Array

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
        print(f'Usage : {sys.argv[0]} <file_name> <num_proc>')
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

def count_characters(freq, buffer, tid, idx, blocks):
    '''
    Function:  count_characters 
    --------------------
    Counts the frequency of characters in an char array
    
    freq: the array that will contain each character's frequency
    buffer: the array that contains the characters
    tid: process id
    idx: start index of process
    blocks: number of blocks
    '''
    stop = tid * blocks + blocks
    for i in range(idx, stop):
        with freq.get_lock():
            freq[buffer[i] - base] += 1

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

    # check arguments, if file name & number of processes are given, return them
    filename, num_proc = check_args()

    # open file (default mode='r')
    f = open_file()

    # get file size
    file_size = obtain_file_size(f)
    
    # print file size
    print(f'File size is {file_size}')

    # frequency vector init with zeros
    freq = np.zeros(N, dtype=np.int32)

    # convert buffer to ascii buffer (numpy array)
    ascii_buff = np.asarray([ord(char) for char in list(f.read(file_size))], dtype=np.int32)

    # create shared frequency vector (shared among processes), must be protected with lock!!!
    shared_freq = Array('i', freq, lock=True)

    # create shared buffer vector (shared among processes), no need for lock (reading-only vector)
    shared_buff = Array('i', ascii_buff, lock=False)

    # create sufficient blocks 
    blocks = int(file_size / num_proc)

    # create processes
    processes = list()
    j = 0
    for i in range(num_proc):
        processes.append(Process(target=count_characters, args=(shared_freq, shared_buff, i, j, blocks,)))
        j += blocks

    # start counting
    start = time.time()

    # start threads
    for i in range(num_proc):
        processes[i].start()

    # wait all threads to finish
    for i in range(num_proc):
        processes[i].join()

    # done
    end = time.time()

    # print char frequency
    [print(f'{j+base} = {shared_freq[j]}') for j in range(N)]

    # print time of execution
    print(f'Time spent for counting: {end-start}')

    # close file
    f.close()
