
import numpy as np
from mpi4py import MPI

TERMINATE = -1
TAG = 200
ROOT = 0
N = 21

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    status = MPI.Status()

    if rank == ROOT:
        begin = MPI.Wtime()
        x=2
        print(f'Primes numbers from 1 to {N} are : ')
        print(x) # Print the first prime 
        for i in range(x+1, N+1):
            if i % x:
                comm.send(obj=i, dest=rank+1, tag=TAG)  # Send primes to P1
        i = TERMINATE
        comm.send(obj=i, dest=rank+1, tag=TAG)  # Send termination signal
        i = comm.recv(source=size-1, tag=TAG, status=status)
        end = MPI.Wtime()

        print(f'\nTime spent for operation: {end-begin} seconds')

    if rank == (size-1):
        x=0
        while True:
            i = comm.recv(source=rank-1, tag=TAG, status=status)
            if not x: 
                x = i # Keep receiving numbers until termination signal is received
            if i == TERMINATE: # If termination signal received OR i is prime
                comm.send(obj=i, dest=ROOT, tag=TAG)
                break
            print(i) 

    if rank != ROOT and rank != (size-1):
        x=0
        while True:
            i = comm.recv(source=rank-1, tag=TAG, status=status)
            if not x: 
                x = i
                print(i)
            if i % x or i == TERMINATE: # If termination signal received OR i is prime
            	comm.send(obj=i, dest=rank+1, tag=TAG)
            if i == TERMINATE:
                break
        
        
