# Makefile for building the CUDA project

# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -g -G

# Target executable
TARGET = framework

# Source files
SRCS = framework.cu

FILES = framework.cu kernel.cu kernel_CPU.C

# Default target
all: $(TARGET)

run: $(TARGET)
	./$(TARGET)

airacuda:
	scp $(FILES) airacuda:/home/u524810/gpu/
	ssh airacuda 'cd /home/u524810/gpu/ && make clean'
	ssh airacuda 'cd /home/u524810/gpu/ && make run'


# Build target
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRCS)

# Clean target
clean:
	rm -f $(TARGET)

.PHONY: all clean