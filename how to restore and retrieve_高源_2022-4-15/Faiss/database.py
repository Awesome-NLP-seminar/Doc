import numpy as np
import time

path = "./database/toy.npy"

shape = (7_500_000, 768)


# 创建
t_begin = time.time()
a_create = np.memmap(path, shape=shape, mode='w+', dtype=np.float32)
t_end = time.time()

print("创建时间:", t_end - t_begin)
print(a_create)
print(a_create.shape)
print("=" * 50)