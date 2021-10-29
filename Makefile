
NVCC           :="/usr/local/cuda/bin/nvcc"
NVCCFLAGS      :=-lineinfo -c -x cu -arch sm_70 -std=c++11 -O3 `pkg-config --cflags opencv4`

all: kernel

kernel: kernel.o
	$(CXX) -o kernel kernel.o -L/usr/local/cuda/lib64 -lcuda -lcudart `pkg-config --libs opencv4`
	
kernel.o: kernel.cu
	$(NVCC) $(NVCCFLAGS)  kernel.cu

clean:
	rm -f kernel kernel.o
