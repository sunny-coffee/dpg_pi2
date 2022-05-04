from unittest import result
import torch
import numpy as np
from masspoint import Masspoint
import matplotlib.pyplot as plt


class DPGPI2Trainer():
    def __init__(self, initPolicy, config_DPG_PI2, learning_rate=1e-3):
        # self.policy = initPolicy
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.learning_rate = learning_rate
        self.batch_size = config_DPG_PI2.batch_size
        self.epoch = config_DPG_PI2.epoch
        self.viapoints = config_DPG_PI2.viapoints
        self.constraints = config_DPG_PI2.constraints
        self.goal = config_DPG_PI2.goal
        self.n_dim = config_DPG_PI2.n_dim
        self.n_reuse = config_DPG_PI2.n_reuse
        self.plotIntermediate = config_DPG_PI2.plotIntermediate 
        self.std = config_DPG_PI2.std
        if config_DPG_PI2.stiffness_learning:
            self.n_dim_kp = config_DPG_PI2.n_dim_kp 
        else:
            self.n_dim_kp = 0 
        self.n_in = config_DPG_PI2.n_dim 
        self.n_out = self.n_dim + self.n_dim_kp 
        self.n_param = initPolicy.theta0.shape[0]
        self.dt = config_DPG_PI2.dt  
        self.n_steps = config_DPG_PI2.duration/config_DPG_PI2.dt 
        self.eliteRolloutsTank = EliteRolloutsTank(self.n_reuse)
        if config_DPG_PI2.model == 'masspoint':
            start = np.array([-10,0.2])
            goal = np.array([0,0])
            self.dynamics_model = Masspoint(1, 1, config_DPG_PI2, start, goal, initPolicy)

    def train(self):
        cost = np.zeros((self.epoch,6))
        for i in range(self.epoch):
            eps = 0
            eval_sample = self.dynamics_model.run_all_steps(eps)
            self.eliteRolloutsTank.add_sample(eval_sample)
            if i == 0:
                init_sample = eval_sample
            if  i%10 == 0:
                print('{}.totCost_eval = {}\n'.format(i, eval_sample.totCost))
                print('{}.transCost_eval = {}\n'.format(i, eval_sample.transCost))
                print('{}.viapointCost_eval = {}\n'.format(i, eval_sample.viapointCost))
                print('{}.accelerationCost_eval = {}\n'.format(i, eval_sample.accelerationCost))
                print('{}.stiffnessCost_eval = {}\n'.format(i, eval_sample.stiffnessCost))
                print('===================================================')
            if i == 0:
                cost[i,:] = [0, eval_sample.totCost, eval_sample.transCost, eval_sample.viapointCost, eval_sample.accelerationCost, eval_sample.stiffnessCost]
            else:
                cost[i,:] = [i*(self.batch_size - self.n_reuse) + self.n_reuse, eval_sample.totCost, eval_sample.transCost, eval_sample.viapointCost, eval_sample.accelerationCost, eval_sample.stiffnessCost]
            samples_epoch = []
            for j in range(len(self.eliteRolloutsTank)):
                sample = self.eliteRolloutsTank[j]
                sample.eps = sample.theta_eps - self.dynamics_model.policy.theta
                samples_epoch.append(sample)

            for k in range(len(self.eliteRolloutsTank), self.batch_size):
                noise_mult = (self.epoch - i)/self.epoch
                noise_mult = max(0.1, noise_mult)
                eps = np.random.randn(self.n_param,1) * noise_mult * self.std
                sample = self.dynamics_model.run_all_steps(eps)
                samples_epoch.append(sample)
            self.updatePIBB(samples_epoch)

        eps = 0
        eval_sample = self.dynamics_model.run_all_steps(eps)
        self.eliteRolloutsTank.add_sample(eval_sample)
        print('{}.totCost_eval = {}\n'.format(self.epoch, eval_sample.totCost))
        print('{}.transCost_eval = {}\n'.format(self.epoch, eval_sample.transCost))
        print('{}.viapointCost_eval = {}\n'.format(self.epoch, eval_sample.viapointCost))
        print('{}.accelerationCost_eval = {}\n'.format(self.epoch, eval_sample.accelerationCost))
        print('{}.stiffnessCost_eval = {}\n'.format(self.epoch, eval_sample.stiffnessCost))
        fin_sample = eval_sample

        # plt.figure()
        # nx = 20
        # ny = 20
        # min_x = -11
        # max_x = 1
        # min_y = -2
        # max_y = 4
        # ax_x = np.linspace(min_x,max_x,nx)
        # ax_y = np.linspace(min_y,max_y,ny)
        # [x_tmp, y_tmp] = np.meshgrid(ax_x,ax_y)
        # # w = np.concatenate((np.reshape(x_tmp,(-1,1)),np.reshape(y_tmp,(-1,1))), axis=1)
        # # wd = np.zeros(w.shape)
        # u = me
        # # print(wd)
        # for i in range(w.shape[0]):
        #     # print(i,w[i,:])            
        #     output = self.dynamics_model.policy.predict(w[i,:],w[i,:],self.dynamics_model.policy.theta)
        #     wd[i,:] = output[0:1]
        # # print(wd)
        # u = np.reshape(wd[:,0],(ny,nx))
        # v = np.reshape(wd[:,1],(ny,nx))
        # plt.streamplot(x_tmp,y_tmp,u,v)
        # plt.show()



        result = Result(cost, init_sample, fin_sample, self.dynamics_model.policy)
        return result

        
        
    def updatePIBB(self,samples):
        cost_list = list(sample.totCost for sample in samples)
        # print(cost_list)
        # print(cost_list)
        max_cost = max(cost_list)
        min_cost = min(cost_list)
        h = 10
        S = np.array(cost_list)
        expS = np.exp(-h*(S - min_cost*np.ones(self.batch_size))/((max_cost-min_cost)*np.ones(self.batch_size)+10e-100))
        P = expS/np.sum(expS)  
        eps_list = list(sample.eps for sample in samples)
        eps_epoch = np.concatenate(eps_list, axis=1)

        dtheta = np.dot(eps_epoch, np.reshape(P,(-1,1)))
        # print(self.dynamics_model.policy.theta)
        # print(dtheta)
        self.dynamics_model.policy.theta = self.dynamics_model.policy.theta + dtheta



class EliteRolloutsTank():
    def __init__(self,n_elite_tank):
        self.max_n_elites = n_elite_tank
        self.buffer = []

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, index):
        return self.buffer[index]

    def add_sample(self,sample):
        self.buffer.append(sample)
        self.buffer = sorted(self.buffer, key=lambda x: x.totCost)
        if len(self.buffer) > self.max_n_elites:
            self.buffer.pop()


class Result():
    def __init__(self, cost, init_sample, fin_sample, policy) -> None:
        self.cost =cost
        self.init_sample = init_sample
        self.fin_sample = fin_sample
        self.policy = policy


 



