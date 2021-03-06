
import cupy as cp
import time
import os

# constants
N = 10000

THREADS_PER_BLOCK = 512
BLOCKS = int((N + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK)
###########

# current directory
CURR_DIR = os.path.dirname(os.path.realpath(__file__))

# open file containing C code
f = open(f'{CURR_DIR}\primes.cu', 'rt')

# load code from source file
loaded_from_source = f'extern "C" {f.read()}'

# load c code
module = cp.RawModule(code=loaded_from_source, backend='nvcc')

# load kernel
ker_primes = module.get_function('find_primes')

# create cupy array (will be the sorted one)
primes_gpu = cp.zeros(N, dtype=cp.int32)

# start timer
begin = time.time()

# perform sort!
ker_primes((BLOCKS,), (THREADS_PER_BLOCK,), (primes_gpu, N))

# move primes array to host
primes_cpu = primes_gpu.get()

# stop timer
end = time.time()

# print sorted array
[print(i) for i in range(2, N) if primes_cpu[i]]

# print time elapsed
print(f'Time elapsed {end-begin}')
