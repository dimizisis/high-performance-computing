
import cupy as cp
import time
import os

# constants
N = 10
LOWER = 0
UPPER = N*4
###########

# current directory
CURR_DIR = os.path.dirname(os.path.realpath(__file__))

# open file containing C code
f = open(f'{CURR_DIR}\count_sort.cu', 'rt')

# load code from source file
loaded_from_source = f'extern "C" {f.read()}'

# load c code
module = cp.RawModule(code=loaded_from_source)

# load kernel
ker_sort = module.get_function('count_sort')

# initialize array (random)
array = cp.random.randint(LOWER, high=UPPER, size=N, dtype=cp.int32)

# print initial array
print(array)

# create cupy array (will be the sorted one)
sorted_array = cp.empty(N, dtype=cp.int32)

# start timer
begin = time.time()

# perform sort!
ker_sort((N,), (1,), (array, sorted_array, N))

# stop timer
end = time.time()

# print sorted array
print(sorted_array)

# print time elapsed
print(f'Time elapsed {end-begin}')
