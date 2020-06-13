
import cupy as cp
import time
import os

# constants
N = 10
LOWER = 0
UPPER = N*4

THREADS_PER_BLOCK = 512
BLOCKS = int((N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK)
###########

# current directory
CURR_DIR = os.path.dirname(os.path.realpath(__file__))

# open file containing C code
f = open(f'{CURR_DIR}\count_sort.cu', 'rt')

# load code from source file
loaded_from_source = f'extern "C" {f.read()}'

# load c code
module = cp.RawModule(code=loaded_from_source, backend='nvcc')

# load kernel
ker_sort = module.get_function('count_sort')

# initialize array (random)
array_gpu = cp.random.randint(LOWER, high=UPPER, size=N, dtype=cp.int32)

# print initial array
print(array_gpu)

# create cupy array (will be the sorted one)
sorted_array_gpu = cp.empty(N, dtype=cp.int32)

# start timer
begin = time.time()

# perform sort!
ker_sort((BLOCKS,), (THREADS_PER_BLOCK,), (array_gpu, sorted_array_gpu, N))

# move sorted array to host
sorted_array_cpu = sorted_array_gpu.get()

# stop timer
end = time.time()

# print sorted array
print(sorted_array_cpu)

# print time elapsed
print(f'Time elapsed {end-begin}')
