import torch
import numpy as np
import matplotlib.pyplot as plt
from noise import NormalActionNoise
from model import Actor, Critic
from buffer import Buffer
from read_matlab_data import read_matlab_data
from configurations import config_DPG_PI2
from demos_dataset import DemosDaataset
from torch.utils.data import DataLoader
from masspoint_DPG import Masspoint
from configurations import config_DPG_PI2

# actor_pretrain_criterion = torch.nn.MSELoss(reduction = 'mean')
# actor_pretrain_optimizer = torch.optim.Adam(actor_model.parameters(),lr=0.001)

# data_path = "./data/sineShape2"
# dt = 0.1
# n_dim =2 
# n_steps = 120
# dataset = DemosDaataset(data_path, dt, n_dim, n_steps)
# # print(dataset[0])
# train_loader = DataLoader(dataset=dataset, batch_size=2, shuffle=True)

# # plt.figure()
# loss_list = []
# for epoch in range(100):
#     losses = []
#     for i,data in enumerate(train_loader):
#         # print(data)
#         state, target = data
#         # print(state.shape)
#         state = np.transpose(state,(1,0,2))
#         target = np.transpose(target,(1,0,2))
#         state = state.to(torch.float32)
#         target = target.to(torch.float32)
#         batch_size = state.shape[1]
# #         for i in range(batch_size):
# #             plt.plot(state[:,i,0], state[:,i,1])
# # plt.show()
#         loss=0
#         actor_pretrain_optimizer.zero_grad()
#         hidden = actor_model.init_hidden(batch_size)
#         # print(state.shape, target.shape)
#         outputs = []
#         for input,label in zip(state,target):
#             hidden, output = actor_model(input,hidden)
#             loss += actor_pretrain_criterion(output,label)

#         loss.backward()
#         actor_pretrain_optimizer.step()
    
#         losses.append(loss.item())
#     loss_list.append(np.mean(losses))

# plt.figure()
# plt.plot(loss_list)
# plt.show()

# torch.save(actor_model, './model/actor_model.pt')



actor_model = torch.load('./model/actor_model.pt')
critic_model = Critic(2, 2, 1)
masspoint = Masspoint(1,1,config_DPG_PI2)
# Learning rate for actor-critic models
critic_lr = 0.0002
actor_lr = 0.00001

critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr = critic_lr)
actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr = actor_lr)

buffer = Buffer(2, 2, 120)

ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []


total_episodes = 200
criterion = torch.nn.MSELoss(reduction = 'mean')
# Takes about 4 min to train
for ep in range(total_episodes):

    state = torch.tensor([[-10,0.2]])

    # actions, cost = actor_model.predict_with_noise(state)
    actions, cost = masspoint.run_all_steps_with_noise(state,actor_model)
    # actions = torch
    # print(actions)
    # print(cost)
    # print(state.shape, actions.shape, cost.shape)
    buffer.record((state, actions, cost))


    state_batch, actions_batch, cost_batch = buffer.sample()

    # print(state_batch.shape)
    # print(actions_batch.shape)
    current_cost_batch = critic_model(state_batch, actions_batch)
    # print(current_cost_batch)
    # print(cost_batch)
    critic_loss = criterion(current_cost_batch, cost_batch)
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    actor_loss = critic_model(state_batch, masspoint.run_all_steps(state_batch,actor_model)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    cost = masspoint.test_all_steps(state,actor_model)

    ep_reward_list.append(cost)
    if ep%20 == 0:
        print("Episode * {} *  Cost is ==> {}".format(ep, cost))

plt.plot(ep_reward_list)
plt.xlabel("Episode")
plt.ylabel("Epsiodic Cost")
plt.show()