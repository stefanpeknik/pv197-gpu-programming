# Makefile for building the CUDA project

# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -g -G -lineinfo

# Target executable
TARGET = framework

# Source files
SRCS = framework.cu

# Default target
all: $(TARGET)

# Build target
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRCS)

# Clean target
clean:
	rm -f $(TARGET)

.PHONY: all clean