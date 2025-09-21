print("================= Platform Info =================")
import platform
print("Python:", platform.python_version())
print("Platform:", platform.platform())
print()

print("============== PyTorch GPU support ==============")
import torch
print("PyTorch:", torch.__version__)
print("CUDA Available:", torch.cuda.is_available())
print(f"Number of GPUs: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"\nDevice {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        print(f"  Capability: {torch.cuda.get_device_capability(i)}")
        print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
        print(f"  Memory Cached: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
print()


print("============== TensorFlow GPU support ==============")
import tensorflow as tf
print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())

print("\nNumber of Physical Device:", len(tf.config.list_physical_devices('GPU')))
print("\nNumber of Logical Device:", len(tf.config.list_logical_devices('GPU')))

from tensorflow.python.client import device_lib
devices = device_lib.list_local_devices()
for i, device in enumerate(devices):
    print(f"\nDevice {i}:")
    print(f"  Name: {device.name}")   
    print(f"  Description: {device.physical_device_desc}")
print()
