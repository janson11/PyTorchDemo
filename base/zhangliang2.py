import torch
import numpy as np

# 1. NumPy 数组转换为PyTorch 张量
print("1. NumPy 数组转换为PyTorch 张量")
numpy_array = np.array([[1, 2, 3], [4, 5, 6]])
print("NumPy 数组:\n", numpy_array)

# 使用torch.from_numpy()将NumPy数组转换为张量
tensor_from_numpy = torch.from_numpy(numpy_array)
print("转换后的PyTorch 张量:\n", tensor_from_numpy)

# 修改NumPy数组，观察张量的变化(共享内存)
numpy_array[0, 0] = 100
print("修改后的NumPy数组：\n", numpy_array)
print("PyTorch张量也会同步变化：\n", tensor_from_numpy)

# 2. PyTorch张量转换为NumPy数组
print("\n2. PyTorch张量转换为NumPy数组")
tensor = torch.tensor([[7, 8, 9], [10, 11, 12]], dtype=torch.float32)
print("PyTorch张量：\n", tensor)

# 使用tensor.numpy()将张量转换为NumPy数组
numpy_from_tensor = tensor.numpy()
print("转换后的NumPy数组\n",numpy_from_tensor)

# 修改张量，观察NumPy数组的变化(共享内存)
tensor[0, 0] = 77
print("修改后的PyTorch张量：\n", tensor)
print("NumPy数组也会同步变化：\n", numpy_from_tensor)


# 3. 注意：不共享内存的情况(需要复制数据)
print("\n3.使用clone() 保证独立数据")
tensor_independent = torch.tensor([[13,14,15],[16,17,18]],dtype=torch.float32)
numpy_independent = tensor_independent.clone().numpy() # 使用clone复制数据
print("原始张量:\n",tensor_independent)
tensor_independent[0,0] = 0 # 修改张量数据
print("修改后的张量:\n",tensor_independent)
print("NumPy数组(不会同步变化):\n",numpy_independent)