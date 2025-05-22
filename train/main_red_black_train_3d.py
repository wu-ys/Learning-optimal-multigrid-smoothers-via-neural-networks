# %% [markdown]
# # Imports

# %%
import os
from importlib import reload
from itertools import product
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from problems import *
from model_train import *
import scipy as sp
import pyamg
device = 'cuda'

# %% [markdown]
# # Setup

# %%
res1 = np.zeros((3,3,3))
res1[0,0,1] = 1/16
res1[2,0,1] = 1/16
res1[0,2,1] = 1/16
res1[2,2,1] = 1/16
res1[1,0,0] = 1/16
res1[1,0,2] = 1/16
res1[1,2,0] = 1/16
res1[1,2,2] = 1/16
res1[0,1,0] = 1/16
res1[2,1,0] = 1/16
res1[0,1,2] = 1/16
res1[2,1,2] = 1/16
res1[1,1,1] = 1/4

res2 = np.zeros((3,3,3))
res2[0,1,1] = 1/12
res2[1,0,1] = 1/12
res2[1,1,0] = 1/12
res2[2,1,1] = 1/12
res2[1,2,1] = 1/12
res2[1,1,2] = 1/12
res2[1,1,1] = 1/4
res2[0,0,0] = 1/32
res2[0,0,2] = 1/32
res2[0,2,0] = 1/32
res2[2,0,0] = 1/32
res2[2,2,0] = 1/32
res2[0,2,2] = 1/32
res2[2,0,2] = 1/32
res2[2,2,2] = 1/32

# %%
def map_3_to_1(grid_size=8):
    # maps 3D coordinates to the corresponding 1D coordinate in the matrix.
    k = np.zeros((grid_size, grid_size, 3, 3))
    M = np.reshape(np.arange(grid_size ** 3), (grid_size, grid_size, grid_size))
    M = np.concatenate([M, M], 0)
    M = np.concatenate([M, M], 1)
    for i in range(3):
        I = (i - 1) % grid_size
        for j in range(3):
            J = (j - 1) % grid_size
            k[:, :, i, j] = M[I:I + grid_size, J:J + grid_size]
    return k
def get_p_matrix_indices_one(grid_size):
    K = map_3_to_1(grid_size=grid_size)
    indices = []
    half_size = grid_size // 2
    for ic in range(half_size):
        i = 2 * ic + 1
        for jc in range(half_size):
            j = 2 * jc + 1
            for kc in range(half_size):
                k = 2 * kc + 1
                J = int(half_size * (half_size * kc + jc) + ic)
                for p in range(3):
                    for q in range(3):
                        for r in range(3):
                            I = int(K[i, j, k, p, q, r])
                            indices.append([I, J])

    return np.array(indices)


def compute_p2(P_stencil, grid_size):
    indexes = get_p_matrix_indices_one(grid_size)
    P = sp.csr_matrix(arg1=(P_stencil.reshape(-1), (indexes[:, 1], indexes[:, 0])),
                   shape=((grid_size//2) ** 2, (grid_size) ** 2))

    return P
def prolongation_fn(grid_size):
    res_stencil = np.double(np.zeros((3,3)))
    k=16
    res_stencil[0,0] = 1/k
    res_stencil[0,1] = 2/k
    res_stencil[0,2] = 1/k
    res_stencil[1,0] = 2/k
    res_stencil[1,1] = 4/k
    res_stencil[1,2] = 2/k
    res_stencil[2,0] = 1/k
    res_stencil[2,1] = 2/k
    res_stencil[2,2] = 1/k
    P_stencils= np.zeros((grid_size//2,grid_size//2,3,3))
    for i in range(grid_size//2):
        for j in range(grid_size//2):
            P_stencils[i,j,:,:]=res_stencil
    return compute_p2(P_stencils, grid_size).astype(np.double)  # imaginary part should be zero
def rotate_idx(size):
    X = []
    Y = []
    for i in range(size):
        for j in range(size):
            if (i+j)%2==0:
                X.append(i)
                Y.append(j)
    new_X = []
    new_Y = []
    for k in range(len(X)):
        i = X[k]
        j = Y[k]
        new_j = (j-i)//2+size//2
        new_i = (i+j)//2
        new_X.append(new_i)
        new_Y.append(new_j)

#     B[new_X,new_Y] = A[X,Y]
    return new_X,new_Y,X,Y

# %%
n = 256
A = torch.load("/cephfs/shared/crh_dataset/svp_4sources/A_12.pt").double().to(device)
f = torch.load("/cephfs/shared/crh_dataset/svp_4sources/b_12.pt").double().to(device).reshape(-1)

mxl = 5
levels = []
def coo_to_tensor(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    shape = coo.shape
    return torch.sparse.DoubleTensor(i, v, torch.Size(shape)).to(device)

initial_u = torch.ones(n*n*n,1, device=device).double()
for i in range(mxl):
    level={}
    level['A'] = A.to_sparse_coo()
    A_coo = A.to_sparse_coo()
    rows, cols = A_coo.indices()
    data = A_coo.values()
    mask = (rows == cols)
    diag_rows = rows[mask]
    diag_data = data[mask]
    D = torch.sparse_coo_tensor(indices = torch.stack([diag_rows, diag_rows],dim=0), values = diag_data, size=A_coo.shape, dtype=A.dtype, device = A.device)
    level['D'] = D
    level['N'] = n
    level['l'] = A.shape[0]
    if i%2==0:
        R = pyamg.gallery.stencil_grid(res1,(n,n,n)).tocsr()
        R = R[0:n*n*n:2,:]
        R = coo_to_tensor(R.tocoo())
#         R = R[:,0:n*n//2+1]
        P = R.T*2
        level['square'] = True
    else:
        R = pyamg.gallery.stencil_grid(res2,(n,n,n)).tocsr()
        #R = pyamg.gallery.stencil_grid(res2,(n,n)).toarray()
        R = R[0:n*n*n:2,:]
        R = R[:,0:n*n*n:2]
        R = coo_to_tensor(R.tocoo())
        level['rotate_idx'] = rotate_idx(n)
        idx = []
        for j in range(n//2+1):
            idx = idx+list(range(j*n,j*n+n//2+1))
        R = R[idx,:]
        P=R.T*2
        n=n//2+1
        level['square'] = False


    A = torch.sparse.mm(A_coo.cpu(), P.cpu())
    Anew = torch.sparse.mm(R.cpu(), A.cpu())

    A = Anew.to(device)

    print(A._nnz)

    level['R']=R.to(device)
    level['P']=P.to(device)
    levels = levels+[level]


# %%
nb_problem_instances = 50
optimizer = 'Adam'
learning_rate = 1e-3
nb_layers = 5
problem_instances1 = [Problem(k=k,levels=levels[3:5],mxl=2) for k in np.random.randint(1,3,nb_problem_instances)]
model1=alphaCNN(
                 batch_size=10,
                 learning_rate=1e-3,
                 max_epochs=4,
                 nb_layers=nb_layers,
                 tol=1e-6,
                 stable_count=10,
                 optimizer=optimizer,
                 random_seed=9,initial = 5,kernel_size=3,initial_kernel=(1/s[1,1]))
model1.fit(problem_instances1)
problem_instances2 = [Problem(k=k,levels=levels[2:5],mxl=3,net_trained=[model1.net]) for k in np.random.randint(1,5,nb_problem_instances)]
model2=alphaCNN(
                 batch_size=10,
                 learning_rate=1e-4,
                 max_epochs=8,
                 nb_layers=nb_layers,
                 tol=1e-6,
                 stable_count=10,
                 optimizer=optimizer,
                 random_seed=9,initial = 5,kernel_size=3,initial_kernel=0.01)
model2.fit(problem_instances2)
problem_instances3 = [Problem(k=k,levels=levels[1:5],mxl=4,net_trained=[model2.net,model1.net]) for k in np.random.randint(1,7,nb_problem_instances)]
model3=alphaCNN(
                 batch_size=10,
                 learning_rate=1e-4,
                 max_epochs=8,
                 nb_layers=nb_layers,
                 tol=1e-6,
                 stable_count=10,
                 optimizer=optimizer,
                 random_seed=9,initial = 5,kernel_size=3,initial_kernel=0.001)
model3.fit(problem_instances3)
problem_instances4 = [Problem(k=k,levels=levels[0:5],mxl=5,net_trained=[model3.net,model2.net,model1.net]) for k in np.random.randint(1,10,nb_problem_instances)]
model4=alphaCNN(
                 batch_size=10,
                 learning_rate=1e-4,
                 max_epochs=4,
                 nb_layers=nb_layers,
                 tol=1e-6,
                 stable_count=10,
                 optimizer=optimizer,
                 random_seed=9,initial = 5,kernel_size=3,initial_kernel=0.001)
model4.fit(problem_instances4)

# %%



