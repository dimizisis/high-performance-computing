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
      * [Shell Sort](#insertion_sort)
      * [Sieve of Eratosthenes](#sieve)
   * [mpi4py (Python)](#mpi4py)
     * [Character Frequency](#char_freq-mpi4py)
     * [Count Sort (Enumeration Sort)](#count_sort-mpi4py)
     * [Insertion Sort](#insertion_sort-mpi4py)
     * [Shell Sort](#insertion_sort-mpi4py)
     * [Sieve of Eratosthenes](#sieve)

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

###### Linux:
```
./dda
```

##### dda_parallel_midpoint.c
```
g++ dda_parallel_midpoint.c -lbgi -lgdi32 -lcomdlg32 -luuid -loleaut32 -lole32 -fopenmp -o dda_p
```

###### Windows
```
dda_p
```

###### Linux:
```
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
```
chaos
```

###### Linux:
```
./chaos
```

##### chaos_parallel.c
```
g++ chaos_parallel.c -lbgi -lgdi32 -lcomdlg32 -luuid -loleaut32 -lole32 -fopenmp -o chaos_p
```

###### Windows
```
chaos_p
```

###### Linux:
```
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
```
char_freq_p
```

###### Linux:
```
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
```
cs_p
```

###### Linux:
```
./cs_p
```

#### Execution Time

Tested both with random array, size 90000.

![alt text](https://i.imgur.com/f1UNxsI.png "count_sort")

### insertion_sort

Parallelized insertion sort algorithm (enumeration sort), using omp parallel and critical block (lock). Change the N macro inside the code. Sequential calculaton is even faster.

#### Usage

```
gcc -fopenmp insertion_sort_parallel.c -o is_p
```

###### Windows
```
is_p
```

###### Linux:
```
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
```
epsilon_parallel
```

###### Linux:
```
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
```
pi_parallel_array
```

###### Linux:
```
./pi_parallel_array
```

#### pi_parallel_atomic

Each thread writes to global variable (protected with atomic operation).

##### Usage

```
gcc -fopenmp pi_parallel_atomic.c -o pi_parallel_atomic
```

###### Windows
```
pi_parallel_atomic
```

###### Linux:
```
./pi_parallel_atomic
```

#### pi_parallel_critical

Each thread writes to global variable (protected with critical section).

##### Usage

```
gcc -fopenmp pi_parallel_critical.c -o pi_parallel_critical
```

###### Windows
```
pi_parallel_critical
```

###### Linux:
```
./pi_parallel_critical
```

#### pi_parallel_critical

Each thread writes to local variable and then adds its result to global variable (pi).

##### Usage

```
gcc -fopenmp pi_parallel_local.c -o pi_parallel_local
```

###### Windows
```
pi_parallel_local
```

###### Linux:
```
./pi_parallel_local
```

#### pi_parallel_reduction

Each thread writes to global variable (using OpenMP's reduction operation).

##### Usage

```
gcc -fopenmp pi_parallel_reduction.c -o pi_parallel_reduction
```

###### Windows
```
pi_parallel_reduction
```

###### Linux:
```
./pi_parallel_reduction
```

#### Execution Time

![alt text](https://i.imgur.com/0PWoykh.png "pi")

As we can observe, reduction works better compared to other methods (pi_parallel_local is reduction, pi_parallel_reduction uses OpenMP's reduction operation).

## MPI
