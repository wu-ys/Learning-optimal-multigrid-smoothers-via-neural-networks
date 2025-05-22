# %%
import numpy as np
import torch
from problems import *
from model_train import *
import scipy as sp
import pyamg
device = 'cuda'

# %%
def diffusion_stencil_2d(epsilon=1.0, theta=0.0, type='FE'):
    eps = float(epsilon)  # for brevity
    theta = float(theta)

    C = np.cos(theta)
    S = np.sin(theta)
    CS = C*S
    CC = C**2
    SS = S**2

    if(type == 'FE'):
        a = (-1*eps - 1)*CC + (-1*eps - 1)*SS + (3*eps - 3)*CS
        b = (2*eps - 4)*CC + (-4*eps + 2)*SS
        c = (-1*eps - 1)*CC + (-1*eps - 1)*SS + (-3*eps + 3)*CS
        d = (-4*eps + 2)*CC + (2*eps - 4)*SS
        e = (8*eps + 8)*CC + (8*eps + 8)*SS

        stencil = np.array([[a, b, c],
                            [d, e, d],
                            [c, b, a]]) / 6.0

    elif type == 'FD':

        a = -0.5*(eps - 1)*CS
        b = -(eps*SS + CC)
        c = -a
        d = -(eps*CC + SS)
        e = 2.0*(eps + 1)

        stencil = np.array([[a+c, d-2*c, 2*c],
                            [b-2*c, e+4*c, b-2*c],
                            [2*c, d-2*c, a+c]])


    return stencil
res1 = np.zeros((3,3))
res1[0,1] = 1/8
res1[1,0] = 1/8
res1[1,1] = 1/2
res1[1,2] = 1/8
res1[2,1] = 1/8

res2 = np.zeros((3,3))
res2[0,0] = 1/8
res2[0,2] = 1/8
res2[1,1] = 1/2
res2[2,0] = 1/8
res2[2,2] = 1/8

# %%
def map_2_to_1(grid_size=8):
    # maps 2D coordinates to the corresponding 1D coordinate in the matrix.
    k = np.zeros((grid_size, grid_size, 3, 3))
    M = np.reshape(np.arange(grid_size ** 2), (grid_size, grid_size)).T
    M = np.concatenate([M, M], 0)
    M = np.concatenate([M, M], 1)
    for i in range(3):
        I = (i - 1) % grid_size
        for j in range(3):
            J = (j - 1) % grid_size
            k[:, :, i, j] = M[I:I + grid_size, J:J + grid_size]
    return k
def get_p_matrix_indices_one(grid_size):
    K = map_2_to_1(grid_size=grid_size)
    indices = []
    for ic in range(grid_size // 2):
        i = 2 * ic + 1
        for jc in range(grid_size // 2):
            j = 2 * jc + 1
            J = int(grid_size // 2 * jc + ic)
            for k in range(3):
                for m in range(3):
                    I = int(K[i, j, k, m])
                    indices.append([I, J])

    return np.array(indices)


def compute_p2(P_stencil, grid_size):
    indexes = get_p_matrix_indices_one(grid_size)
    P = sp.sparse.csr_matrix(arg1=(P_stencil.reshape(-1), (indexes[:, 1], indexes[:, 0])),
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
n = 33
mxl = 5
levels=[]
s = diffusion_stencil_2d(100,np.pi/12,'FD')*2
A = pyamg.gallery.stencil_grid(s, (n,n)).tocsr()
def coo_to_tensor(coo):
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.DoubleTensor(values)
    shape = coo.shape
    return torch.sparse.DoubleTensor(i, v, torch.Size(shape)).to(device)
x = np.random.rand(n*n,1)
f = A*x
f = torch.from_numpy(f)
x = torch.from_numpy(x)
initial_u = torch.ones(n*n,1).double()
for i in range(mxl):
    level={}
    level['A'] = coo_to_tensor(A.tocoo())
    D = sp.sparse.diags(1/sp.sparse.csr_matrix.diagonal(A))
    level['D'] = coo_to_tensor(D.tocoo())
    level['N'] = n
    level['l'] = A.shape[0]
    print(A.shape)
    if i%2==0:
        R = pyamg.gallery.stencil_grid(res1,(n,n)).tocsr()
        R = R[0:n*n:2,:]
#         R = R[:,0:n*n//2+1]
        P = R.T*2
        level['square'] = True
    else:
        R = pyamg.gallery.stencil_grid(res2,(n,n)).tocsr()
        #R = pyamg.gallery.stencil_grid(res2,(n,n)).toarray()
        R = R[0:n*n:2,:]
        R = R[:,0:n*n:2]
        level['rotate_idx'] = rotate_idx(n)
        idx = []
        for j in range(n//2+1):
            idx = idx+list(range(j*n,j*n+n//2+1))
        R = R[idx,:]
        P=R.T*2
        n=n//2+1
        level['square'] = False
    A = R@A@P
    R = coo_to_tensor(R.tocoo())
    P = coo_to_tensor(P.tocoo())

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



