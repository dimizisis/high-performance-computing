
import sys
from os import SEEK_END, SEEK_SET
import numpy as np
import sys
from os import SEEK_END
import cupy as cp
import time

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

def load_c_code():
    '''
    Function:  load_c_code 
    --------------------
    Loads C code, which contains kernel

    Returns code as str
    '''
    import os

    # current directory
    CURR_DIR = os.path.dirname(os.path.realpath(__file__))

    # open file containing C code
    f = open(f'{CURR_DIR}\count_characters.cu', 'rt')

    return f'extern "C" {f.read()}'

if __name__ == '__main__':

    # check arguments, if file name is given return it
    filename = check_args()

    # open file (default mode='r')
    f = open_file()

    # get file size
    file_size = obtain_file_size(f)
    
    # print file size
    print(f'File size is {file_size}\n')

    # convert buffer to ascii buffer (cupy array)
    ascii_buff = cp.asarray([ord(char) for char in list(f.read(file_size))], dtype=cp.int32)

    # frequency vector init with zeros
    freq = cp.zeros(N, dtype=cp.int32)

    # load code from source file
    loaded_from_source = load_c_code()

    # load c code
    module = cp.RawModule(code=loaded_from_source)

    # load kernel
    ker_count = module.get_function('count_characters')

    # start counting
    start = time.time()

    # count characters!
    ker_count((N,), (1,), (ascii_buff, freq, file_size, base))

    # done
    end = time.time()

    # print char frequency
    [print(f'{j+base} = {freq[j]}') for j in range(N)]

    # print time of execution
    print(f'Time spent for counting: {end-start}')

    # close file
    f.close()
