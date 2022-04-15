import time

import numpy as np

d = 128                           # dimension
nb = 7500000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 10000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 10000.

k = 10                          # we want to see 10 nearest neighbors


import faiss                     # make faiss available

nlist = 100
quantizer = faiss.IndexFlatL2(d)  # the other index
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# here we specify METRIC_L2, by default it performs inner-product search

# make it an IVF GPU index
co = faiss.GpuMultipleClonerOptions()
co.shard = True # 是否共享数据
print("to gpu")
gpu_index_ivf = faiss.index_cpu_to_gpus_list(index_ivf, gpus=[0, 1, 2, 3], co=co)

assert not gpu_index_ivf.is_trained
gpu_index_ivf.train(xb)        # add vectors to the index
assert gpu_index_ivf.is_trained

gpu_index_ivf.add(xb)          # add vectors to the index
print(gpu_index_ivf.ntotal)


gpu_index_ivf.nprobe = 10
t0 = time.time()
D, I = gpu_index_ivf.search(xq, k)  # actual search
t1 = time.time()
print(I[-5:])                  # neighbors of the 5 last queries

print(f"search time:{t1-t0}")