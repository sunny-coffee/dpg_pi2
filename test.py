from masspoint import Masspoint_one_step
import numpy as np
from network import Actor, Critic
import torch



a = tensor([[16.1198],
        [16.1198],
        [16.1194],
        [16.1198],
        [16.1198],
        [16.1194],
        [16.1194],
        [16.1194],
        [16.1194],
        [16.1194],
        [16.1194],
        [16.1194],
        [16.1194],
        [16.1194],
        [16.1198],
        [16.1198],
        [16.1194],
        [16.1198],
        [16.1194],
        [16.1194],
        [16.1194],
        [16.1198],
        [16.1194],
        [16.1194],
        [16.1194],
        [16.1194],
        [16.1194],
        [16.1198],
        [16.1194],
        [16.1198],
        [16.1198],
        [16.1198],
        [16.1198],
        [16.1194],
        [16.1194],
        [16.1194],
        [16.1198],
        [16.1198],
        [16.1194],
        [16.1198],
        [16.1194],
        [16.1198],
        [16.1194],
        [16.1198],
        [16.1194],
        [16.1198],
        [16.1198],
        [16.1198],
        [16.1194],
        [16.1194],
        [16.1194],
        [16.1198],
        [16.1198],
        [16.1198],
        [16.1198],
        [16.1194],
        [16.1198],
        [16.1198],
        [16.1198],
        [16.1194],
        [16.1194],
        [16.1198],
        [16.1194],
        [16.1194]])

    
# rnn = torch.nn.RNNCell(2,4)
# hidden = torch.zeros(2)
# input = torch.zeros(4)

# z = rnn(input,hidden)


# actor = Actor(2, 4, 120)
# init_state = torch.tensor([[-10,0.2],[-9,0.3]])
# actions, cost = actor.predict_with_noise(init_state)
# print(actions.shape)
# # print(actions.shap
# print(cost)


# critic = Critic(2,2,1)
# value = critic(init_state, actions)
# print(value)







# import numpy as np
# import matplotlib.pyplot as plt

# nx = 20
# ny = 20
# min_x = 1
# max_x = 9
# min_y = 1
# max_y = 9
# ax_x = np.linspace(min_x,max_x,nx)
# ax_y = np.linspace(min_y,max_y,ny)
# [x_tmp, y_tmp] = np.meshgrid(ax_x,ax_y)
# w = np.concatenate((np.reshape(x_tmp,(-1,1)),np.reshape(y_tmp,(-1,1))), axis=1)
# wd = np.zeros(w.shape)
# # print(wd)
# for i in range(w.shape[0]):
#     # print(i,w[i,:])            
#     # output = self.predict(w[i,:],w[i,:],self.theta0)
#     wd[i,0] = np.sin(w[i,0])
#     wd[i,1] = np.sin(w[i,1])
# print(wd)
# u = np.reshape(wd[:,0],(ny,nx))
# v = np.reshape(wd[:,1],(ny,nx))
# plt.streamplot(x_tmp,y_tmp,u,v)
# plt.show()