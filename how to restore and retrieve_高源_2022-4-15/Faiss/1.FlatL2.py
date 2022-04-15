import time

import numpy as np

d = 128                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 10000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 10000.

import faiss                   # make faiss available
index = faiss.IndexFlatL2(d)   # build the index
print(index.is_trained)
index.add(xb)                  # add vectors to the index
print(index.ntotal)
print("==================================")
k = 10                          # we want to see 10 nearest neighbors
D, I = index.search(xb[:5], k) # sanity check
print("D:", D.shape, "I:", I.shape)
print(I)
print(D)
print("==================================")
t0 = time.time()
D, I = index.search(xq, k)     # actual search
t1 = time.time()

print("D:", D.shape, "I:", I.shape)
print(I[:5])                   # neighbors of the 5 first queries
print(I[-5:])                  # neighbors of the 5 last queries

print(f"search time:{t1-t0}")