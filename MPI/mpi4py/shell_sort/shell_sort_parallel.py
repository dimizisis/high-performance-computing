
import numpy as np
from mpi4py import MPI

TAG = 200
ROOT = 0
N = 10
LOWER = 0
UPPER = N*2

def calculate_displacements(sendcounts):
    '''
    Function:  calculate_displacements 
    --------------------
    Calculates displacements (for MPI_Scatterv) and puts them on an array
    
    sendcounts: pointer of the array which contains the size of all local arrays of each process (size of the array is the number of processes)
    '''
    return np.insert(np.cumsum(sendcounts), 0, 0)[0:-1]

def calculate_sendcounts(split):
    '''
    Function:  calculate_sendcounts 
    --------------------
    Calculates sendcounts (how many elements we're going to send to each process, for MPI_Scatterv) and puts them on an array
 
    split: pointer of the array which contains the split of initial array (split is done using numpy's split)
    '''
    return [len(split[i]) for i in range(len(split))]

def merge(x, n1, y, n2):
    '''
    Function:  merge 
    --------------------
    Merges two arrays with different length

    x: pointer of the first array
    n1: size of array x
    y: pointer of the second array
    n2: size of array y
    
    Returns: pointer of the merged array
    '''
    i=0
    j=0
    k=0
    result = np.empty(dtype=int, shape=n1+n2)
    while i < n1 and j < n2:
        if x[i] < y[j]:
            result[k] = x[i]
            i += 1
            k += 1
        else:
            result[k] = y[j]
            j += 1
            k += 1
    if i == n1:
        while j < n2:
            result[k] = y[j]
            j += 1
            k += 1
    else:
        while i < n1:
            result[k] = x[i]
            i += 1
            k += 1

    return result

def shell_sort(array, n):
    '''
    Function:  shell_sort 
    --------------------
    Sorts an integer array, using the shell sort algorithm

    a: the array that will be sorted
    n: number of elements in the array
    '''
    h = int(n/2)
    while h > 0:
        for i in range (h, n):
            t = array[i]
            j=i
            while j >= h and t < array[j-h]:
                array[j] = array[j - h]
                j -= h
            array[j] = t
        h = int(h/2)
    
def calculate_count(sendcounts, rank):
    sum = 0
    for i in range(rank+1):
        sum += sendcounts[i]
    return sum

if __name__ == '__main__':

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()
    status = MPI.Status()

    array = np.empty(N, np.int)
    if rank == ROOT:
        array = np.random.randint(low=LOWER, high=UPPER, size=N)
        # root makes the split
        split = np.array_split(array, size)
        print(f'\nInitial Array: {array}')
        begin = MPI.Wtime()
        my_chunk = split[rank]
        # Every process has split[rank] chunk :-))
        for i in range(1,size):
            comm.send(list(split[i]), i, TAG)   # Root sends to each process
    else:
        my_chunk = comm.recv(source=ROOT, tag=TAG, status=status)   # The others receive

    # Each thread will have different local_locations vector 
    local_array = np.asarray(my_chunk)

    # Perform Shell Sort only on the local array with size chunk (every process)
    shell_sort(local_array, local_array.size)

    # Each thread will have different local_locations vector 
    if rank == ROOT:
        # Root process will only send to the next its array
        comm.send(obj=local_array.size, dest=rank+1, tag=TAG)
        comm.Send(buf=local_array, dest=rank+1, tag=TAG)

    else:  # if process is not the root

        # Receive the size of rank-1 array
        recv_count = comm.recv(source=rank-1, tag=TAG, status=status)
        
        # Make a recv array object
        recv_array = np.empty(dtype=np.int, shape=recv_count)

        # Receive array from previous process
        comm.Recv(buf=recv_array, source=rank-1, tag=TAG, status=status)

        # Merge your array with the received one (merged_array)
        merged_array = merge(local_array, local_array.size, recv_array, recv_count)

        comm.send(obj=merged_array.size, dest=(rank+1)%size, tag=TAG)
        # Send the merged array to the next process (when last is last process turn, will send array to root because of modulo)
        comm.Send(buf=merged_array, dest=(rank+1)%size, tag=TAG)

    # Receive & print results 
    if rank == ROOT:
        # Root receives merged_array from last process (size == N)
        recv_count = comm.recv(source=size-1, tag=TAG, status=status)
        # Results vector
        results = np.empty(shape=recv_count, dtype=np.int)
        # Receive final vector
        comm.Recv(buf=results, source=size-1, tag=TAG, status=status)
        end = MPI.Wtime()
        print(f'\nSorted Array: {results}')
        print(f'\nTime elapsed for operation: {end-begin} seconds.\n')