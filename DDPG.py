import torch
import numpy as np
import matplotlib.pyplot as plt
from noise import NormalActionNoise
from network import Actor, Critic
from buffer import Buffer


actor_model = Actor(2, 4, 120)
critic_model = Critic(2, 2, 1)

# Learning rate for actor-critic models
critic_lr = 0.002
actor_lr = 0.001

critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr = critic_lr)
actor_optimizer = torch.optim.Adam(actor_model.parameters(), lr = actor_lr)

buffer = Buffer(2, 2, 120)

ep_reward_list = []
# To store average reward history of last few episodes
avg_reward_list = []


total_episodes = 2000
criterion = torch.nn.MSELoss(reduction = 'mean')
# Takes about 4 min to train
for ep in range(total_episodes):

    state = torch.tensor([[-10,0.2],[2,2]])

    actions, cost = actor_model.predict_with_noise(state)
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

    actor_loss = critic_model(state, actor_model(state)).mean()
    actor_optimizer.zero_grad()
    actor_loss.backward()
    actor_optimizer.step()

    ep_reward_list.append(cost)
    avg_reward = np.mean(ep_reward_list[-40:])
    if ep%50 == 0:
        print("Episode * {} * Avg Reward is ==> {}".format(ep, avg_reward))
    avg_reward_list.append(avg_reward)

plt.plot(avg_reward_list)
plt.xlabel("Episode")
plt.ylabel("Avg. Epsiodic Reward")
plt.show()