import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import patches

def get_dynamics_on_grid(A, mins, maxs, step = 0.25):
    Xs = [np.arange(i, j+step, step) for i,j in zip(mins, maxs)]
    grid = np.meshgrid(*Xs)
    loc = np.c_[[i.flatten() for i in grid]]
    dXdt = A@loc
    dXdt_grid = [i.reshape(j.shape) for i,j in zip(dXdt, grid)]
    return grid, dXdt_grid

def solve_trajectory(A, x0, T):
    # continuous time solution
    x = []
    for t in T:
        x_i = torch.matrix_exp(torch.tensor(A, dtype=torch.float) * t) # solution to LDS is the matrix exponential
        x_i = (x_i@torch.tensor(x0, dtype=torch.float).reshape(-1,1)).flatten()
        x.append(x_i.numpy())
    traj = np.c_[x]
    return traj

def rotation_matrix(theta):
    k = torch.tensor([1.0,0,0]) # axis of rotation
    k /= torch.linalg.norm(k)

    K = np.array([[    0,  -k[2],  k[1]],
                  [k[2],     0,  -k[0]],
                  [-k[1], k[0],    0]], dtype=np.float64)

    # Rodrigues formula
    return np.sin(theta)*K + K@K*(1-np.cos(theta))+np.eye(3) 


# PLOTTTING

def plot_3D_dictionary(As, bs, s, title, rslds):

    fig = plt.figure(figsize=(3,3*4),dpi=150)
    T_pred = np.linspace(0, 35, 501)
    for i, (a, b) in enumerate(zip(As, bs)):
        ax = fig.add_subplot(4,1,i+1, projection='3d')
        ax.view_init(25, 45)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.zaxis.set_rotate_label(False)
        ax.set_zlabel(rf'$f_{i+1}$ ', rotation=0)

        if rslds:
            for x0 in s[:-1]:
                operator_path = [x0.flatten()]
                for i in T_pred:
                    x0_ = operator_path[-1][:,None]
                    x1_hat = (a @ x0_ + b[:,None]).flatten()
                    operator_path.append(x1_hat)
                operator_path = np.c_[operator_path]
                ax.plot(*operator_path.T, 'steelblue', alpha=0.5)

            x0 = s[-1]
            operator_path = [x0.flatten()]
            for i in T_pred:
                x0_ = operator_path[-1][:,None]
                x1_hat = (a @ x0_ + b[:,None]).flatten()
                operator_path.append(x1_hat)
            operator_path = np.c_[operator_path]

        else:
            
            for x0 in s[:-1]:
                operator_path = solve_trajectory(a.numpy(), x0, T_pred) 
                ax.plot(*operator_path.T, 'steelblue', alpha=0.5)


            x0 = s[-1]
            operator_path = solve_trajectory(a.numpy(), x0, T_pred) 


        ax.plot(*operator_path.T, 'red', alpha=0.5)

        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        ax.set_zlim(-2,2)


    fig.suptitle(title, y=0.92)
    plt.subplots_adjust(hspace=0.1)
    plt.show()
    
    
def plot_interpolation_base(title, T_pred, n_plots = 7):
    fig = plt.figure(figsize=(15,3),dpi=150)
    axs = []
    for i in range(n_plots):
        ax = fig.add_subplot(1,n_plots,i+1, projection='3d')
        ax.view_init(25, 45)
        ax.grid(False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])
        ax.zaxis.set_rotate_label(False)

        ax.set_xlim(-2,2)
        ax.set_ylim(-2,2)
        ax.set_zlim(-2,2)

        axs.append(ax)

    fig.suptitle(title, y=0.8)


    arrow = patches.ConnectionPatch(
        [0.1,-0.1],
        [-0.08,-0.1],
        coordsA=axs[0].transData,
        coordsB=axs[-1].transData,
        # Default shrink parameter is 0 so can be omitted
        color="black",
        arrowstyle="<|-|>",  # "normal" arrow
        mutation_scale=10,  # controls arrow head size
        linewidth=2,
    )
    fig.patches.append(arrow)
    plt.subplots_adjust(wspace=0.0)
    return fig, axs

