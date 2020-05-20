import cupy as cp
import time
import os

# constants
N = 10000
###########

# current directory
CURR_DIR = os.path.dirname(os.path.realpath(__file__))

# open file containing C code
f = open(f'{CURR_DIR}\primes.cu', 'rt')

# load code from source file
loaded_from_source = f'extern "C" {f.read()}'

# load c code
module = cp.RawModule(code=loaded_from_source)

# load kernel
ker_primes = module.get_function('find_primes')

# create cupy array (will be the sorted one)
primes = cp.zeros(N, dtype=cp.int32)

# start timer
begin = time.time()

# perform sort!
ker_primes((N,), (1,), (primes, N))

# stop timer
end = time.time()

# print sorted array
[print(i) for i in range(2, N) if primes[i]]

# print time elapsed
print(f'Time elapsed {end-begin}')
