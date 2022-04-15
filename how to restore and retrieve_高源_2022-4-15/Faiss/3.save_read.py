import numpy as np
import time
d = 128                           # dimension
nb = 100000                      # database size
nq = 10000                       # nb of queries
np.random.seed(1234)             # make reproducible
xb = np.random.random((nb, d)).astype('float32')
xb[:, 0] += np.arange(nb) / 10000.
xq = np.random.random((nq, d)).astype('float32')
xq[:, 0] += np.arange(nq) / 10000.

import faiss

# nlist = 100
# k = 10
# quantizer = faiss.IndexFlatL2(d)  # the other index
# index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# # here we specify METRIC_L2, by default it performs inner-product search
#
# assert not index.is_trained
# index.train(xb)
# assert index.is_trained
#
# index.add(xb)                  # add may be a bit slower as well
#
output_path = "./database/ivf.index"
# faiss.write_index(index, output_path)

index = faiss.read_index(output_path)
print(index.ntotal)