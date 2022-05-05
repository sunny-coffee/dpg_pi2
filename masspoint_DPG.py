import numpy as np
import torch

class Masspoint():
    def __init__(self, mass, damp, config):
        self.mass = mass
        self.damp = damp
        
        self.dt = config.dt
        self.goal = config.goal
        self.n_dim = config.n_dim
        self.n_steps = int(config.duration/config.dt)
        self.stiffness_learning = config.stiffness_learning
        self.constraints = config.constraints
        self.exsit_force_field = config.exsit_force_field
        self.n_dim_kp = config.n_dim_kp
        self.viapoints = config.viapoints
        self.constraints = config.constraints
        self.action_size = 2
        

    def run_all_steps_with_noise(self, init_state, actor_model):
        batch_size = init_state.shape[0]
        w_epoch = np.zeros((self.n_steps, batch_size, self.n_dim))
        wd_epoch = np.zeros((self.n_steps, batch_size, self.n_dim))
        wdd_epoch = np.zeros((self.n_steps, batch_size, self.n_dim))
        control_parameter_epoch = np.zeros((self.n_steps, batch_size, self.n_dim_kp))

        with torch.no_grad():
            actions = torch.zeros(self.n_steps, batch_size, self.action_size)
            w = init_state
            wd = torch.zeros_like(w)
            ref_w = w
            hidden = actor_model.init_hidden(batch_size)
            for n in range(self.n_steps):
                hidden, output = actor_model(ref_w, hidden)
                ref_wd = output[:, 0:2]
                kp = output[:, 2:]
                kd = 2*torch.sqrt(kp)
                ref_w = ref_w + ref_wd * self.dt
                ref_w = (ref_w + 0.01*torch.randn_like(ref_w)).clip(-4, 4)
                force = 0
                action = kp * (ref_w-w) + kd * (ref_wd-wd)
                # action = (action + 0.01*torch.randn_like(action)).clip(-5, 5)
                actions[n,:,:] = action

                [w,wd,wdd] = self.run_one_step(w,wd,action,force) 
                w_epoch[n] = w.numpy() 
                wd_epoch[n] = wd.numpy()  
                wdd_epoch[n] = wdd.numpy()  
                control_parameter_epoch[n, :] = kp.numpy() 
            
            totCost, transCost, viapointCost, accelerationCost, stiffnessCost = self.compute_cost(w_epoch, wd_epoch, wdd_epoch, control_parameter_epoch)
        return actions, totCost
    
    def run_all_steps(self, init_state, actor_model):
        batch_size = init_state.shape[0]
        actions = torch.zeros(self.n_steps, batch_size, self.action_size)
        w = init_state
        wd = torch.zeros_like(w)
        ref_w = w
        hidden = actor_model.init_hidden(batch_size)
        for n in range(self.n_steps):
            hidden, output = actor_model(ref_w, hidden)
            ref_wd = output[:, 0:2]
            kp = output[:, 2:]
            kd = 2*torch.sqrt(kp)
            ref_w = ref_w + ref_wd * self.dt
            force = 0
            action = kp * (ref_w-w) + kd * (ref_wd-wd)
            actions[n,:,:] = action
            [w,wd,wdd] = self.run_one_step(w,wd,action,force)
        return actions

    
    def test_all_steps(self, init_state, actor_model):
        batch_size = init_state.shape[0]
        w_epoch = np.zeros((self.n_steps, batch_size, self.n_dim))
        wd_epoch = np.zeros((self.n_steps, batch_size, self.n_dim))
        wdd_epoch = np.zeros((self.n_steps, batch_size, self.n_dim))
        control_parameter_epoch = np.zeros((self.n_steps, batch_size, self.n_dim_kp))
        
        with torch.no_grad():
            w = init_state
            wd = torch.zeros_like(w)
            ref_w = w
            hidden = actor_model.init_hidden(batch_size)
            for n in range(self.n_steps):
                hidden, output = actor_model(ref_w, hidden)
                ref_wd = output[:, 0:2]
                kp = output[:, 2:]
                kd = 2*torch.sqrt(kp)
                ref_w = ref_w + ref_wd * self.dt
                force = 0
                action = kp * (ref_w-w) + kd * (ref_wd-wd)
                [w,wd,wdd] = self.run_one_step(w,wd,action,force) 
                w_epoch[n] = w.numpy() 
                wd_epoch[n] = wd.numpy()  
                wdd_epoch[n] = wdd.numpy()  
                control_parameter_epoch[n, :] = kp.numpy() 
            totCost, transCost, viapointCost, accelerationCost, stiffnessCost = self.compute_cost(w_epoch, wd_epoch, wdd_epoch, control_parameter_epoch)
        return totCost
    

    def run_one_step(self, w, wd, action, fieldforce):
        # print(kp,ref_w,kd,ref_wd)
        # print(kp.shape,ref_w.shape,kd.shape,ref_wd.shape)
        wdd = (action - wd * self.damp + fieldforce)/self.mass
        wd = wdd * self.dt + wd
        w = wd * self.dt + w
        return w, wd, wdd


    def compute_field_force(self, position, gain):
        nPoints = 100
        forceFieldTraj = np.zeros((nPoints, 2))
        for cnt1 in range(nPoints):
            forceFieldTraj[cnt1, 1] = -10* (1 - cnt1/nPoints)
            forceFieldTraj[cnt1, 2] = np.sin(2*np.pi*(1 - cnt1/nPoints))

        distances = np.linalg.norm(position - forceFieldTraj, axis=1)
        minDist = np.min(distances)
        minIndex = np.where(distances == minDist)

        if minIndex != nPoints-1:
            tang = forceFieldTraj[minIndex+1, :] - forceFieldTraj[minIndex, :]
        else:
            tang = forceFieldTraj[minIndex, :] - forceFieldTraj[minIndex-1, :]
            
        force = np.dot(tang/np.linalg.norm(tang), gain * minDist* np.array([[0, 1], [-1, 0]]))
        if np.dot(force, (position - forceFieldTraj[minIndex, :])) < 0:
            force = -force

        return force

    def compute_cost(self, w, wd, wdd, control_parameter_epoch):

        batch_size = w.shape[1]
        viapointCost = np.zeros(batch_size)
            
        transCostSteps = np.linalg.norm(wd, axis=2)
        accelerationCostSteps  =  np.linalg.norm(wdd, axis=2)
        stiffnessCostSteps = np.linalg.norm(control_parameter_epoch, axis=2)

        for i in range(self.viapoints.shape[0]):
            distances = np.linalg.norm(w - self.viapoints[i, :], axis=2)
            minDist = np.min(distances, axis=0)
            viapointCost += minDist
        d_final  =  np.linalg.norm(w[-1] - self.goal, axis=1)
        viapointCost += d_final

        viapointCostWeight = 1
        stiffnessCostWeight = 0.00001
        accelerationCostWeight = 0.0012
        transCostWeight = 0.001

        viapointCost = viapointCost * viapointCostWeight
        stiffnessCost = np.sum(stiffnessCostSteps, axis=0) * stiffnessCostWeight
        accelerationCost = np.sum(accelerationCostSteps, axis=0) * accelerationCostWeight
        transCost = np.sum(transCostSteps, axis=0) * transCostWeight

        totCost = viapointCost + accelerationCost + stiffnessCost + transCost
        # print(totCost.shape)
        return totCost, transCost, viapointCost, accelerationCost, stiffnessCost

