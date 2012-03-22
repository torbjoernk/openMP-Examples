# GENERAL PARAMETERS (tune here)
OPT	= -O3
C_OMP	= -fopenmp
DEBUG	= -ggdb3 -O0
STD	= -Wall -std=c++0x

# COMPILER FLAGS
CFLAGS	= $(STD)
LDFLAGS	= $(OPT) $(DEBUG)

# TARGETS
all: vector_add dotprod_omp matxvec_sparse

vector_add: vector_add.cpp
	g++ $(CFLAGS) $(LDFLAGS) $(C_OMP) -o vector_add vector_add.cpp

dotprod_omp: dotprod_omp.cpp
	g++ $(CFLAGS) $(LDFLAGS) $(C_OMP) -o dotprod_omp dotprod_omp.cpp

matxvec_sparse: matxvec_sparse.cpp
	g++ $(CFLAGS) $(LDFLAGS) $(C_OMP) -o matxvec_sparse matxvec_sparse.cpp 