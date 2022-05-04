import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from masspoint import Masspoint_one_step
from noise import NormalActionNoise
from configurations import config_DPG_PI2

class Actor(nn.Module):
    def __init__(self, state_size:int, output_size:int, n_steps:int, hidden_size:int=128):
        """Initialize."""
        super(Actor, self).__init__()

        self.dt = 0.1
        self.n_steps = n_steps
        self.output_size = output_size
        self.action_size = 2
        # self.noise
        self.masspoint = Masspoint_one_step(1,1)

        self.state2hidden1 = nn.Linear(state_size, hidden_size)
        # self.state2hidden2 = nn.Linear(hidden_size*4, hidden_size)

        self.grucell = nn.RNNCell (input_size=state_size, hidden_size=hidden_size, bias=True)

        self.output = nn.Linear(hidden_size, hidden_size)
        self.output2refwd = nn.Linear(hidden_size, 2)
        self.output2kp = nn.Linear(hidden_size, 2)
        # self.output2.weight.data.uniform_(-init_w, init_w)
        # self.output2.bias.data.uniform_(-init_w, init_w)

    def forward(self, init_state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        #  init_state: (batch_size, state_size)
        #  action: (batch_size, action_size)
        #  actions: (src_len, batch_size, action_size)
        init_hidden = self.state2hidden1(init_state)
        hidden = F.relu(init_hidden)
        # init_hidden = self.state2hidden2(init_hidden)
        # hidden = F.relu(init_hidden)

        batch_size = init_state.size(0)
        actions = torch.zeros(self.n_steps, batch_size, self.action_size)
        w = init_state
        wd = torch.zeros_like(w)
        ref_w = w
        for n in range(self.n_steps):
            hidden = self.step(ref_w, hidden)
            output = self.output(hidden)
            output = F.relu(output)

            ref_wd = self.output2refwd(output)
            ref_wd = F.tanh(ref_wd) * 2

            kp = self.output2kp(output)
            kp = F.sigmoid(kp) * 30

            # output = self.output2kp(output)
            # ref_wd = output[:,0:2]

            ref_w = ref_w + ref_wd * self.dt 
            # print(output)
            # kp =  output[:, 2:]
            # kp = torch.maximum(kp, torch.zeros_like(kp)) 
            kd = 2*torch.sqrt(kp)
            force = 0
            action = kp * (ref_w-w) + kd * (ref_wd-wd)
            [w,wd,wdd] = self.masspoint.run_one_step(w,wd,action,force) 
            actions[n,:,:] = action
        return actions

    def step(self, state:torch.Tensor, hidden:torch.Tensor) -> torch.Tensor:
        #  state: (batch_size, state_size)
        #  hidden: (batch_size, hidden_size)
        # print(state.shape)
        # print(hidden.shape)
        next_hidden = self.grucell(state,hidden)
        return next_hidden

    def predict_with_noise(self, init_state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        #  init_state: (batch_size, state_size)
        #  action: (batch_size, action_size)
        #  actions: (src_len, batch_size, action_size)
        n_dim = config_DPG_PI2.n_dim
        n_dim_kp = config_DPG_PI2.n_dim_kp
        batch_size = init_state.shape[0]
        w_epoch = np.zeros((self.n_steps, batch_size, n_dim))
        wd_epoch = np.zeros((self.n_steps, batch_size, n_dim))
        wdd_epoch = np.zeros((self.n_steps, batch_size, n_dim))
        control_parameter_epoch = np.zeros((self.n_steps, batch_size, n_dim_kp))

        with torch.no_grad():
            init_hidden = self.state2hidden1(init_state)
            hidden = F.relu(init_hidden)
            # init_hidden = self.state2hidden2(init_hidden)
            # hidden = F.relu(init_hidden)

            batch_size = init_state.size(0)
            actions = torch.zeros(self.n_steps, batch_size, self.action_size)
            w = init_state
            wd = torch.zeros_like(w)
            ref_w = w
            for n in range(self.n_steps):
                # hidden = self.step(ref_w, hidden)
                # output = self.output1(hidden)
                # output = F.relu(output)
                # output = self.output2(output)
                # ref_wd = output[:,0:2] 
                # # print(output)
                # kp =  output[:, 2:]
                # kp = torch.maximum(kp, torch.zeros_like(kp)) 
                hidden = self.step(ref_w, hidden)
                output = self.output(hidden)
                output = F.relu(output)

                ref_wd = self.output2refwd(output)
                ref_wd = F.tanh(ref_wd) * 2

                kp = self.output2kp(output)
                kp = F.sigmoid(kp) * 30

                kd = 2*torch.sqrt(kp)
                ref_w = ref_w + ref_wd * self.dt
                force = 0
                action = kp * (ref_w-w) + kd * (ref_wd-wd)
                action = (action + 0.1*torch.randn_like(action)).clip(-5, 5)
                actions[n,:,:] = action

                [w,wd,wdd] = self.masspoint.run_one_step(w,wd,action,force) 
                w_epoch[n] = w.detach().numpy() 
                wd_epoch[n] = wd.detach().numpy()  
                wdd_epoch[n] = wdd.detach().numpy()  
                control_parameter_epoch[n, :] = kp.detach().numpy() 
            # print(w_epoch)
            totCost, transCost, viapointCost, accelerationCost, stiffnessCost = self.compute_cost(w_epoch, wd_epoch, wdd_epoch, control_parameter_epoch)

        return actions, totCost
    
    def predict_without_noise(self, init_state: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        #  init_state: (batch_size, state_size)
        #  action: (batch_size, action_size)
        #  actions: (src_len, batch_size, action_size)
        n_dim = config_DPG_PI2.n_dim
        n_dim_kp = config_DPG_PI2.n_dim_kp
        batch_size = init_state.shape[0]
        w_epoch = np.zeros((self.n_steps, batch_size, n_dim))
        wd_epoch = np.zeros((self.n_steps, batch_size, n_dim))
        wdd_epoch = np.zeros((self.n_steps, batch_size, n_dim))
        control_parameter_epoch = np.zeros((self.n_steps, batch_size, n_dim_kp))
        with torch.no_grad():
            init_hidden = self.state2hidden1(init_state)
            hidden = F.relu(init_hidden)
            # init_hidden = self.state2hidden2(init_hidden)
            # hidden = F.relu(init_hidden)

            batch_size = init_state.size(0)
            actions = torch.zeros(self.n_steps, batch_size, self.action_size)
            w = init_state
            wd = torch.zeros_like(w)
            ref_w = w
            for n in range(self.n_steps):
                # hidden = self.step(ref_w, hidden)
                # output = self.output1(hidden)
                # output = F.relu(output)
                # output = self.output2(output)
                # ref_wd = output[:,0:2] 
                # # print(output)
                # kp =  output[:, 2:]
                # kp = torch.maximum(kp, torch.zeros_like(kp)) 
                hidden = self.step(ref_w, hidden)
                output = self.output(hidden)
                output = F.relu(output)

                ref_wd = self.output2refwd(output)
                ref_wd = F.tanh(ref_wd) * 2

                kp = self.output2kp(output)
                kp = F.sigmoid(kp) * 30

                kd = 2*torch.sqrt(kp)
                ref_w = ref_w + ref_wd * self.dt
                force = 0
                action = kp * (ref_w-w) + kd * (ref_wd-wd)
                actions[n,:,:] = action

                [w,wd,wdd] = self.masspoint.run_one_step(w,wd,action,force) 
                w_epoch[n] = w.detach().numpy() 
                wd_epoch[n] = wd.detach().numpy()  
                wdd_epoch[n] = wdd.detach().numpy()  
                control_parameter_epoch[n, :] = kp.detach().numpy() 
        
            totCost, transCost, viapointCost, accelerationCost, stiffnessCost = self.compute_cost(w_epoch, wd_epoch, wdd_epoch, control_parameter_epoch)

        return actions, totCost

    def compute_cost(self, w, wd, wdd, control_parameter_epoch):
        # print(w.shape)
        # print(w[20,:])
        batch_size = w.shape[1]
        # transCostSteps = np.zeros(self.n_steps, batch_size)
        # accelerationCostSteps = np.zeros(self.n_steps, batch_size)
        # stiffnessCostSteps = np.zeros(self.n_steps, batch_size)
        viapointCost = np.zeros(batch_size)

        # for n in range(self.n_steps):
            
        transCostSteps = np.linalg.norm(wd, axis=2)
        accelerationCostSteps  =  np.linalg.norm(wdd, axis=2)
        stiffnessCostSteps = np.linalg.norm(control_parameter_epoch, axis=2)

        for i in range(config_DPG_PI2.viapoints.shape[0]):
            distances = np.linalg.norm(w - config_DPG_PI2.viapoints[i, :], axis=2)
            minDist = np.min(distances, axis=0)
            viapointCost += minDist
        d_final  =  np.linalg.norm(w[-1] - config_DPG_PI2.goal, axis=1)
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


class Critic(nn.Module):
    def __init__(self, state_size:int, action_size:int, value_size:int, hidden_size:int=128):
        super(Critic, self).__init__()
        self.state2hidden1 = nn.Linear(state_size, hidden_size)
        # self.state2hidden2 = nn.Linear(state_size*4, hidden_size)
        self.grucell = nn.RNN(input_size=action_size, hidden_size=hidden_size, num_layers=1, bias=True, bidirectional=False)
        self.output = nn.Linear(hidden_size, value_size)

    def forward(self, init_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        #  init_state: (batch_size,state_size)
        #  actions: (src_len, batch_size, input_size)
        init_hidden = self.state2hidden1(init_state)
        init_hidden = F.relu(init_hidden)
        # init_hidden = self.state2hidden2(init_hidden)
        # init_hidden = F.relu(init_hidden)

        # print(actions.shape, init_hidden.shape)
        init_hidden = torch.unsqueeze(init_hidden, 0)
        output, h_n = self.grucell(actions, init_hidden)
        h_n = torch.squeeze(h_n)
        x = F.relu(h_n)
        value = self.output(x) 

        return value