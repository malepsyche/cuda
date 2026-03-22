CUDA_HOME ?= /usr/local/cuda
NVCC      ?= $(CUDA_HOME)/bin/nvcc
TARGET    ?= wave2d_solver
SRC       ?= wave2d_solver.cu

ARCH      ?= sm_80
CXXFLAGS  ?= -O3 -std=c++17 -lineinfo
LDFLAGS   ?= -lcusparse -lstdc++fs

all: $(TARGET)

$(TARGET): $(SRC)
	$(NVCC) $(CXXFLAGS) -arch=$(ARCH) $< -o $@ $(LDFLAGS)

run-global: $(TARGET)
	./$(TARGET) --method global --output-dir output/global

run-shared: $(TARGET)
	./$(TARGET) --method shared --block-x 16 --block-y 16 --output-dir output/shared

run-cusparse: $(TARGET)
	./$(TARGET) --method cusparse --output-dir output/cusparse

clean:
	rm -rf $(TARGET) output results

.PHONY: all clean run-global run-shared run-cusparse
