# Makefile for building the CUDA project

# Compiler
NVCC = nvcc

# Compiler flags
CFLAGS = -g -G


# Target executable
TARGET = framework

# Source files
SRCS = framework.cu

FILES = framework.cu kernel.cu kernel_CPU.C Makefile

# Default target
all: $(TARGET)

run: clean $(TARGET)
	./$(TARGET)
	./$(TARGET)

release: clean $(TARGET)_release
	./$(TARGET)
	./$(TARGET)

airacuda-cp:
	scp $(FILES) airacuda:/home/u524810/gpu/

barracuda-cp:
	scp $(FILES) barracuda:/home/u524810/gpu/

# Build target
$(TARGET): $(SRCS)
	$(NVCC) $(CFLAGS) -o $(TARGET) $(SRCS)

# Build release target
$(TARGET)_release: $(SRCS)
	$(NVCC) -o $(TARGET) $(SRCS)

# Clean target
clean:
	rm -f $(TARGET)

.PHONY: all clean
