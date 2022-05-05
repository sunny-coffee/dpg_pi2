import torch
from torch.utils.data import Dataset
import scipy.io as scio
from scipy import interpolate
import numpy as np

class DemosDaataset(Dataset):
    def __init__(self, data_path, dt, n_dim,n_steps):
        self.dt = dt
        self.n_dim = n_dim
        data = scio.loadmat(data_path)
        demos = data.get('demos')
        demo_list = list(demo.T for demo in demos.ravel())

        # x = np.arange(0, 2*np.pi+np.pi/4, 2*np.pi/8)
        # y = np.sin(x)
        # tck = interpolate.splrep(x, y, s=0)
        # xnew = np.arange(0, 2*np.pi, np.pi/50)
        # ynew = interpolate.splev(xnew, tck, der=0)

        for i,demo in enumerate(demo_list):
            # real_steps = demo.shape[0]
            new_steps = np.linspace(0, demo.shape[0]-1, n_steps)
            real_steps = np.arange(0, demo.shape[0])
            tck_x = interpolate.splrep(real_steps, demo[:,0], s=0)
            tck_y = interpolate.splrep(real_steps, demo[:,1], s=0)
            xnew = interpolate.splev(new_steps, tck_x, der=0)
            ynew = interpolate.splev(new_steps, tck_y, der=0)
            # demo = np.zeros()
            demo = np.column_stack((xnew,ynew))
            vel = np.diff(demo, axis=0)/dt
            vel = np.insert(vel, 0, np.zeros(n_dim), axis=0)
            kp = np.ones_like(demo) * 20
            demo_list[i] = np.concatenate((demo,vel,kp), axis=1)
        self.demo_list = demo_list

    def __getitem__(self, index):
        return self.demo_list[index][:,0:self.n_dim], self.demo_list[index][:,self.n_dim:]

    def __len__(self):
        return len(self.demo_list)