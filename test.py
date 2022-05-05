from masspoint import Masspoint_one_step
import numpy as np
from network import Actor, Critic
import torch
import matplotlib.pyplot as plt

# print(np.linspace(0,5,3))
# print(np.arange(0,5))
actor_model = torch.load('./model/actor_model.pt')
w_list = []
w = torch.tensor([[-10,0.2]])
w_list.append(w.detach().numpy())
ref_w = w
wd = torch.tensor([[0,0]])
hidden = actor_model.init_hidden(1)
for i in range(120):
    hidden, output = actor_model(w,hidden)
    # print(output)
    ref_wd = output[:, 0:2]
    ref_w = ref_w + ref_wd * 0.1
    kp = output[:, 2:]
    kd = 2*torch.sqrt(kp)
    action = kp * (ref_w-w) + kd * (ref_wd-wd)
    wdd = (action - wd * 1 )/1
    wd = wdd * 0.1 + wd
    w = wd * 0.1 + w
    w_list.append(w.detach().numpy())
w_list = np.concatenate(w_list)
print(w_list)
plt.figure()
plt.plot(w_list[:,0], w_list[:,1])
plt.show()
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