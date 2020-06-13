
import cupy as cp
import time
import os

# constants
N = 10
LOWER = 0
UPPER = N

THREADS_PER_BLOCK = 512
BLOCKS = int((N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK)
###########

# current directory
CURR_DIR = os.path.dirname(os.path.realpath(__file__))

# open file containing C code
f = open(f'{CURR_DIR}/search.cu', 'rt')

# load code from source file
loaded_from_source = f'extern "C" {f.read()}'

# load c code
module = cp.RawModule(code=loaded_from_source, backend='nvcc')

# load kernel
ker_search = module.get_function('lsearch')

# initialize array (random)
array_gpu = cp.random.randint(low=LOWER, high=UPPER, size=N, dtype=cp.int32)

# print initial array
print(array_gpu)

# number to search (key)
key = 2

# index of number
index_gpu = cp.array([-1], dtype=cp.int32)

# start timer
begin = time.time()

# perform search!
ker_search((BLOCKS,), (THREADS_PER_BLOCK,), (array_gpu, N, key, index_gpu))

# move index to host
index_cpu = index_gpu.get()[0]

# stop timer
end = time.time()

# print index
print(f'Num {key} is at index {index_cpu}')

# print time elapsed
print(f'Time elapsed {end-begin}')
