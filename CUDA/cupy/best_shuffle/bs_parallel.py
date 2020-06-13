
import sys
import cupy as cp
import time

NUM_OF_ARGS = 2

THREADS_PER_BLOCK = 512

def get_input_word():
    '''
    Function:  get_input_word 
    --------------------
    Checks if the number of arguments is correct and returns input word (by user)
    '''
    # if no word given, exit
    if len(sys.argv) != NUM_OF_ARGS:
        print(f'Usage : {sys.argv[0]} <input_word>\n')
        exit(1)
    return str(sys.argv[1])

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
    f = open(f'{CURR_DIR}/best_shuffle.cu', 'rt')

    return f'extern "C" {f.read()}'

if __name__ == '__main__':

    # check arguments, if word is given return it
    word = get_input_word()

    n = len(word)

    # t is the device constant array of word (ascii)
    t_gpu = cp.asarray([ord(char) for char in list(word)], dtype=cp.int32)

    # r is the device array, which will contain final word (ascii)
    r_gpu = t_gpu

    # diff is an one-element array, which will containt the difference between final & initial word
    diff_gpu = cp.zeros(1, dtype=cp.int32)

    # load code from source file
    loaded_from_source = load_c_code()

    # load c code
    module = cp.RawModule(code=loaded_from_source, backend='nvcc')

    # load kernel
    ker_shuffle = module.get_function('best_shuffle')

    # start counting
    start = time.time()
    
    # Create sufficient blocks 
    blocks = int((n + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK)

    # count characters!
    ker_shuffle((blocks,), (THREADS_PER_BLOCK,), (t_gpu, r_gpu, diff_gpu, n))
    
    # copy r array to host
    r_cpu = r_gpu.get()

    # copy r array to host
    diff_cpu = diff_gpu.get()

    # done
    end = time.time()

    # print initial word
    print(f'Word after shuffle: {word}')

    # print shuffled word
    print('Word after shuffle: ', end='')
    print(''.join(chr(i) for i in r_cpu))

    # print difference
    print(f'diff: {diff_cpu[0]}')

    # print time of execution
    print(f'Time spent for counting: {end-start}')
