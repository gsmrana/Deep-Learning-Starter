print("============== PyTorch Performance Check ==============")

import time
import torch

# Define the size of the matrices
size = 10000 # Large size for performance comparison

# Create random matrices on CPU
matrix1_cpu = torch.randn(size, size)
matrix2_cpu = torch.randn(size, size)

# Measure time on CPU
print("Executing in CPU... ", end="")
start_time = time.time()
result_cpu = torch.matmul(matrix1_cpu, matrix2_cpu) # on CPU
cpu_time = time.time() - start_time
print(f"Time:{cpu_time: .4f}s")

# Check GPU availability
if not torch.cuda.is_available():
    print("No GPU available!")
    exit()

# Move matrices to GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
matrix1_gpu = matrix1_cpu.to(device)
matrix2_gpu = matrix2_cpu.to(device)

# Measure time on GPU
print("Executing in GPU... ", end="")
start_time = time.time()
result_gpu = torch.matmul(matrix1_gpu, matrix2_gpu) # on GPU
torch.cuda.synchronize() # Ensure all GPU operations are complete
gpu_time = time.time() - start_time
print(f"Time:{gpu_time: .4f}s")

# Compare results
ratio = cpu_time / gpu_time
print(f"Speedup Ratio (CPU/GPU): {ratio:.02f}")
