
NVCC        = nvcc
NVCC_FLAGS  = -O3 -I/usr/local/cuda/include
LD_FLAGS    = -lcudart -L/usr/local/cuda/lib64
EXE	        = dotprod_problem1_CUDA
OBJ	        = dotprod_problem1_CUDA.o support.o

default: $(EXE)

dotprod_problem1_CUDA.o: dotprod_problem1_CUDA.cu kernel.cu support.h
	$(NVCC) -c -o $@ dotprod_problem1_CUDA.cu $(NVCC_FLAGS)

support.o: support.cu support.h
	$(NVCC) -c -o $@ support.cu $(NVCC_FLAGS)

$(EXE): $(OBJ)
	$(NVCC) $(OBJ) -o $(EXE) $(LD_FLAGS)

clean:
	rm -rf *.o $(EXE)

send: 
	scp /Users/miguelbarrios/Documents/School/PDNProgramming/hw/HW4_GPU/Problem_1/gpu/dotprod_problem1_CUDA.cu oucs4473007@schooner.oscer.ou.edu:/home/oucs4473007/hw/hw4/problem1/gpu/

