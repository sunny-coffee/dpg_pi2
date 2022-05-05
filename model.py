import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from masspoint import Masspoint_one_step
from noise import NormalActionNoise
from configurations import config_DPG_PI2

class Actor(nn.Module):
    def __init__(self, state_size:int, output_size:int, n_steps:int, hidden_size:int=128, init_w=0.003):
        """Initialize."""
        super(Actor, self).__init__()

        self.dt = 0.1
        self.n_steps = n_steps
        self.output_size = output_size
        self.action_size = 2
        self.hidden_size = hidden_size
        # self.noise
        self.masspoint = Masspoint_one_step(1,1)

        self.state2hidden1 = nn.Linear(state_size, hidden_size)
        # self.state2hidden2 = nn.Linear(hidden_size*4, hidden_size)

        self.grucell = nn.RNNCell (input_size=state_size, hidden_size=hidden_size, bias=True)

        self.output = nn.Linear(hidden_size, hidden_size)
        self.output2refwd = nn.Linear(hidden_size, 2)
        self.output2kp = nn.Linear(hidden_size, 2)
        self.output2refwd.weight.data.uniform_(-init_w, init_w)
        self.output2refwd.bias.data.uniform_(-init_w, init_w)
        self.output2kp.weight.data.uniform_(-init_w, init_w)
        self.output2kp.bias.data.uniform_(-init_w, init_w)

    def forward(self, state:torch.Tensor, hidden:torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        #  init_state: (batch_size, state_size)
        #  action: (batch_size, action_size)
        #  actions: (src_len, batch_size, action_size)
        # state = state.to(torch.float32)
        hidden = self.grucell(state,hidden)
        output = self.output(hidden)
        output = torch.relu(output)

        ref_wd = self.output2refwd(output)
        ref_wd = torch.tanh(ref_wd) * 4

        kp = self.output2kp(output)
        kp = torch.sigmoid(kp) * 40
        output = torch.cat((ref_wd,kp) , dim=1)
        # print(ref_wd.shape, kp.shape, output.shape)
        return hidden, output

    def init_hidden(self, batch_size):
        return torch.zeros(batch_size, self.hidden_size)


class Critic(nn.Module):
    def __init__(self, state_size:int, action_size:int, value_size:int, hidden_size:int=256):
        super(Critic, self).__init__()
        self.state2hidden = nn.Linear(state_size, hidden_size)
        # self.state2hidden2 = nn.Linear(state_size*4, hidden_size)
        self.grucell = nn.RNN(input_size=action_size, hidden_size=hidden_size, num_layers=1, bias=True, bidirectional=False)
        self.output = nn.Linear(hidden_size, value_size)

    def forward(self, init_state: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        #  init_state: (batch_size,state_size)
        #  actions: (src_len, batch_size, input_size)
        init_hidden = self.state2hidden(init_state)
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