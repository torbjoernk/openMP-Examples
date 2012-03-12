# GENERAL PARAMETERS (tune here)
OPT	= -O
C_OMP	= -fopenmp
STD	= -std=c++0x

# COMPILER FLAGS
CFLAGS	= $(STD)
LDFLAGS	= $(OPT)

# TARGETS
vector_add: vector_add.cpp
	g++ -o vector_add vector_add.cpp $(LDFLAGS) $(C_OMP)

dotprod_omp: dotprod_omp.cpp
	g++ -o dotprod_omp dotprod_omp.cpp $(LDFLAGS) $(C_OMP)
