# High Performance Computing

University projects using OpenMP, MPI (C & Python), Cuda

## Table of Contents
<!--ts-->
   * [General Requirements](#general-requirements)
   * [OpenMP](#openmp)
      * [DDA](#dda)
      * [Chaos](#chaos)
      * [Character Frequency](#char_freq)
      * [Count Sort (Enumeration Sort)](#count_sort)
      * [Insertion Sort](#insertion_sort)
      * [Epsilon](#epsilon)
      * [Pi](#pi)
   * [MPI (C-Lang)](#mpi)
      * [Character Frequency](#char_freq-mpi)
      * [Count Sort (Enumeration Sort)](#count_sort-mpi)
      * [Insertion Sort](#insertion_sort-mpi)
      * [Shell Sort](#shell_sort)
      * [Sieve of Eratosthenes](#sieve)
   * [mpi4py](#mpi4py)
      * [Usage](#usage)
   * [Cuda](#cuda)
      * [Best Shuffle](#best_shuffle)
      * [Character Frequency](#char_freq-cuda)
      * [Count Sort](#count_sort-cuda)
      * [Linear Search](#linear_search)
      * [Primes](#primes) 
      * [String Matching](#string_matching)
   * [CuPy](#cuda)
      * [Best Shuffle](#best_shuffle-cupy)
      * [Character Frequency](#char_freq-cupy)
      * [Count Sort](#count_sort-cupy)
      * [Linear Search](#linear_search-cupy)
      * [Primes](#primes-cupy)

<!--te-->

## General Requirements

* gcc & g++ compilers
* MPI, Cuda libraries

## OpenMP

### DDA

2 Threads running in parallel, one starting from the initial start point of the line (red line), the other starting from the end point (green line).

#### Extra Instructions

In order to run & see output in your cmd (Windows) or command line (Linux) you need Borland Graphics Interface (if running on Windows) or SDL graphics library if running on Linux system. See instructions for [Windows](https://www.cs.colorado.edu/~main/bgi/dev-c++/) & [Linux](https://askubuntu.com/questions/525051/how-do-i-use-graphics-h-in-ubuntu). 

#### Usage:

##### dda_seq.c
```
g++ dda_seq.c -lbgi -lgdi32 -lcomdlg32 -luuid -loleaut32 -lole32 -o dda
```

###### Windows
```
dda
```

###### Linux
```shell
./dda
```

##### dda_parallel_midpoint.c
```
g++ dda_parallel_midpoint.c -lbgi -lgdi32 -lcomdlg32 -luuid -loleaut32 -lole32 -fopenmp -o dda_p
```

###### Windows
```cmd
dda_p
```

###### Linux
```shell
./dda_p
```

##### Screenshot

![alt text](https://i.imgur.com/lbx36hV.png "dda")

### Chaos

Used omp parallel for in draw triangle's for loop. Parallel point drawing. See more about [Chaos Game](https://en.wikipedia.org/wiki/Chaos_game)

#### Extra Instructions

In order to run & see output in your cmd (Windows) or command line (Linux) you need Borland Graphics Interface (if running on Windows) or SDL graphics library if running on Linux system. See instructions for [Windows](https://www.cs.colorado.edu/~main/bgi/dev-c++/) & [Linux](https://askubuntu.com/questions/525051/how-do-i-use-graphics-h-in-ubuntu). 

#### Usage:

##### chaos_seq.c
```
g++ chaos_seq.c -lbgi -lgdi32 -lcomdlg32 -luuid -loleaut32 -lole32 -fopenmp -o chaos
```

###### Windows
```cmd
chaos
```

###### Linux
```shell
./chaos
```

##### chaos_parallel.c
```
g++ chaos_parallel.c -lbgi -lgdi32 -lcomdlg32 -luuid -loleaut32 -lole32 -fopenmp -o chaos_p
```

###### Windows
```cmd
chaos_p
```

###### Linux:
```shell
./chaos_p
```

##### Screenshot

![alt text](https://i.imgur.com/63IPbCN.png "chaos_game")

### char_freq

Counts the frequency of each ASCII character in a txt file (as input).
Parallelized with 7 different ways

* Using Global Array of Locks
* Using Local Array of Locks
* Using Atomic Operation (Global)
* Using Atomic Operation (Local)
* Using Critical Operation (Global)
* Using Critical Operation (Local)
* Using Reduction

#### Usage

```
gcc -fopenmp char_freq_parallel_<method>.c -o char_freq_p
```

###### Windows
```cmd
char_freq_p
```

###### Linux
```shell
./char_freq_p
```

#### Execution Time

Using as input the bible.txt file.
![alt text](https://i.imgur.com/ePObqe9.png "char_freq")

### count_sort

Parallelized count sort algorithm (enumeration sort), using omp parallel for. Change the N macro inside the code.

#### Usage

```
gcc -fopenmp count_sort_parallel.c -o cs_p
```

###### Windows
```cmd
cs_p
```

###### Linux
```shell
./cs_p
```

#### Execution Time

Tested both with random array, size 90000.

![alt text](https://i.imgur.com/f1UNxsI.png "count_sort")

### insertion_sort

Parallelized insertion sort algorithm, using omp parallel and critical block (lock). Change the N macro inside the code. Sequential calculaton is even faster.

#### Usage

```
gcc -fopenmp insertion_sort_parallel.c -o is_p
```

###### Windows
```cmd
is_p
```

###### Linux
```shell
./is_p
```

#### Execution Time

Tested both with random array, size 500000.

![alt text](https://i.imgur.com/I0vMMFg.png "insertion_sort")

As you can see, no real difference in time. This is most likely due to critical block, each thread waits every time to get the lock.

### epsilon

Parallelized calculation of epsilon (in mathematics). Used omp parallel reduction.

#### Usage

```
gcc -fopenmp epsilon_parallel.c -o epsilon_p
```

###### Windows
```cmd
epsilon_parallel
```

###### Linux
```shell
./epsilon_parallel
```

#### Execution Time

![alt text](https://i.imgur.com/qhKtDYR.png "epsilon")

As you can see, no real difference in time. Sequential calculaton is even faster.

### pi

#### pi_parallel_array

Every thread writes to different element of a global array. When all threads are finished (barrier), add all elements' values to variable.

##### Usage

```
gcc -fopenmp pi_parallel_array.c -o pi_parallel_array
```

###### Windows
```cmd
pi_parallel_array
```

###### Linux
```shell
./pi_parallel_array
```

#### pi_parallel_atomic

Each thread writes to global variable (protected with atomic operation).

##### Usage

```
gcc -fopenmp pi_parallel_atomic.c -o pi_parallel_atomic
```

###### Windows
```cmd
pi_parallel_atomic
```

###### Linux
```shell
./pi_parallel_atomic
```

#### pi_parallel_critical

Each thread writes to global variable (protected with critical section).

##### Usage

```
gcc -fopenmp pi_parallel_critical.c -o pi_parallel_critical
```

###### Windows
```cmd
pi_parallel_critical
```

###### Linux
```shell
./pi_parallel_critical
```

#### pi_parallel_critical

Each thread writes to local variable and then adds its result to global variable (pi).

##### Usage

```
gcc -fopenmp pi_parallel_local.c -o pi_parallel_local
```

###### Windows
```cmd
pi_parallel_local
```

###### Linux
```shell
./pi_parallel_local
```

#### pi_parallel_reduction

Each thread writes to global variable (using OpenMP's reduction operation).

##### Usage

```
gcc -fopenmp pi_parallel_reduction.c -o pi_parallel_reduction
```

###### Windows
```cmd
pi_parallel_reduction
```

###### Linux
```shell
./pi_parallel_reduction
```

#### Execution Time

![alt text](https://i.imgur.com/0PWoykh.png "pi")

As we can observe, reduction works better compared to other methods (pi_parallel_local is reduction, pi_parallel_reduction uses OpenMP's reduction operation).

## MPI

### C

#### char_freq-mpi

SPMD: Each thread takes a specific slice of the file of characters and stores the character frequency to local array (each process has its own frequency array). 

Using the MPI_Reduce function (with MPI_SUM operation), all local frequency arrays sums to root's (process 0) total frequency array.

```c
/* Make the reduce */
MPI_Reduce(freq, total_freq, N, MPI_INT, MPI_SUM, ROOT, MPI_COMM_WORLD);
```

![alt text](https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/mpi_reduce_1.png "MPI_Reduce")

Source: https://mpitutorial.com/tutorials/mpi-reduce-and-allreduce/

##### Usage

```shell
mpicc char_freq_parallel_Reduce.c -o char_freq
```

```shell
mpirun -np 4 char_freq // 4 cores
```

#### Execution Time

![alt text](https://i.imgur.com/moj5i1c.png "CharFreq")

#### count_sort-mpi

##### Reduce

SPMD: Each process is aware of the whole array and sorts a specific part of the array (start, stop). Using the MPI_Reduce function (with MPI_SUM operation), each process returns its results to sorted_array.

###### Usage

```shell
mpicc count_sort_parallel_Reduce.c -o cs_r
```

```shell
mpirun -np 4 cs_r // 4 cores
```

##### Locations

Each process has a local array which contains indexes on where the ith element (i=0,1,...,N) should be put, in order the final array to be sorted. For example, if local_locations[2] == 0, then the element in position 2 of the initial array should be put in position zero.

```shell
mpicc count_sort_parallel_Locations.c -o cs_l
```

```shell
mpirun -np 4 cs_l // 4 cores
```

#### Execution Time

All tests with N = 400000.

![alt text](https://i.imgur.com/l5K3pWW.png "Count_Sort_Reduce")

#### insertion_sort-mpi

Parallelized insertion sort algorithm, using pipeline. N is standard (the number of processors).

#### Usage

```shell
mpicc insertion_sort_parallel_Pipeline.c -o is
```

```shell
mpirun -np 4 is // 4 cores
```

#### shell_sort

Parallelized shell sort algorithm, using the method of [Computer Science and Engineering, University at Buffalo](https://cse.buffalo.edu/faculty/miller/Courses/CSE633/prasad-salvi-Spring-2017-CSE633.pdf).

![alt text](https://i.imgur.com/d4fkRBC.png)

##### Usage

```shell
mpicc shell_sort_parallel_Reduce.c -o shell
```

```shell
mpirun -np 4 shell // 4 cores
```

##### Execution Time

![alt_text](https://i.imgur.com/qaDHqsp.png "Shell Sort")

#### sieve

Parallelized Sieve of Eratosthenes algorithm for finding prime numbers in specific range. Used Pipeline (2 alternatives, first is slow, for big N it actually never ends), global array (bitmap alike) and MPI_Scan directive.

###### Pipeline (Alternative 1)

##### Usage

```shell
mpicc sieve_parallel_Pipeline.c -o sieve_pipe
```

```shell
mpirun -np 4 sieve_pipe // 4 cores
```

###### Pipeline (Alternative 2)

##### Usage

```shell
mpicc sieve_parallel_Pipeline_2.c -o sieve_pipe_2
```

```shell
mpirun -np 4 sieve_pipe_2 // 4 cores
```

###### Global Locations

##### Usage

```shell
mpicc sieve_parallel_Locations_Global.c -o sieve_g
```

```shell
mpirun -np 4 sieve_g // 4 cores
```

###### MPI Scan

##### Usage

```shell
mpicc sieve_parallel_Locations_MPI_Scan.c -o sieve_scan
```

```shell
mpirun -np 4 sieve_scan // 4 cores
```

##### Execution Time

Tested for N = 200 (below).

![alt_text](https://i.imgur.com/OfTqaGd.png "Sieve")

### mpi4py

All code from MPI is written in Python, too, using mpi4py lib.

#### Usage For all programs:

```shell
python <name_of_py_file>
```
