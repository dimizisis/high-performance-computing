
import sys
from os import SEEK_END
import numpy as np
from mpi4py import MPI

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

    # check arguments, if file name is given return it
    filename = check_args()

    # open file (default mode='r')
    f = open_file()

    # get file size
    file_size = obtain_file_size(f)
    
    print(f'File size is {file_size}\n')

	# copy the file into the buffer
    buffer = f.read(file_size)

    # frequency vector init with zeros
    freq = np.zeros(N, dtype=np.int)

    # start counting
    start = MPI.Wtime()

    # count characters of buff (of file)
    count_characters(freq, buffer, file_size)

    # done
    end = MPI.Wtime()

    # print char frequency and time of execution
    [print(f'{j+base} = {freq[j]}') for j in range(N)]
    print(f'Time spent for counting: {end-start}')

    # close file
    f.close()
