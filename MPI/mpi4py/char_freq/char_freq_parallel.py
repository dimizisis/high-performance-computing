
import sys
from os import SEEK_END, SEEK_SET
import numpy as np
from mpi4py import MPI

ROOT = 0
NUM_OF_ARGS = 2
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
        print(f'Usage : {sys.argv[0]} <file_name>\n')
        exit(1)
    return sys.argv[1]

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

def count_characters(freq, buffer, file_size):
    '''
    Function:  count_characters 
    --------------------
    Counts the frequency of characters in an char array
    
    freq: the array that will contain each character's frequency
    buffer: the array that contains the characters
    file_size: size of buffer
    '''
    for i in range(file_size):
    	freq[ord(buffer[i]) - base] += 1

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

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    # check arguments, if file name is given return it
    filename = check_args()

    # open file (default mode='r')
    f = open_file()

    # total freq (global)
    total_freq = np.zeros(N, dtype=np.int)

    # get file size
    file_size = obtain_file_size(f)
    
    if rank == ROOT:
        print(f'File size is {file_size}\n')

    # These initialization will be done by all processes
    freq = np.zeros(N, dtype=np.int)
    chunk = int(file_size / size)
    extra = file_size % size
    start = rank * chunk
    stop = start + chunk
    if rank == size-1: stop += extra

    local_file_size = stop - start

	# copy the file into the buffer
    buffer = f.read(local_file_size)

    # seek from the beggining of start
    f.seek(start, SEEK_SET)

    # copy part of the file into the buffer
    buffer = f.read(local_file_size)

    # frequency vector init with zeros
    freq = np.zeros(N, dtype=np.int)

    # start counting
    start = MPI.Wtime()

    # count characters of buff (of file)
    count_characters(freq, buffer, local_file_size)

    # done
    if rank == ROOT:
        end = MPI.Wtime()

    comm.Reduce(freq, total_freq, MPI.SUM, ROOT)

    if rank == ROOT:
        # print char frequency and time of execution
        [print(f'{j+base} = {total_freq[j]}') for j in range(N)]
        print(f'Time spent for counting: {end-start}')

    # close file
    f.close()
