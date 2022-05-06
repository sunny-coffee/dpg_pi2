import numpy as np
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from model import Critic
import matplotlib.pyplot as plt

class BufferDaataset(Dataset):
    def __init__(self):
        self.state = np.load('./data/state.npy')
        self.actions = np.load('./data/actions.npy')
        self.cost = np.load('./data/cost.npy')

    def __getitem__(self, index):
        return self.state[index], self.actions[index], self.cost[index]

    def __len__(self):
        return self.state.shape[0]


dataset = BufferDaataset()
train_loader = DataLoader(dataset=dataset, batch_size=8, shuffle=True)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
critic_model = Critic(2, 2, 1)
critic_model.to(device)
critic_lr = 0.02
critic_optimizer = torch.optim.Adam(critic_model.parameters(), lr = critic_lr)
criterion = torch.nn.MSELoss(reduction = 'mean')

loss_list = []
for epoch in range(100):
    losses = []
    for i,data in enumerate(train_loader):
        # print(data)
        state, actions,cost = data
        # print(state.shape)
        # print(state.shape, actions.shape, cost.shape)
        actions = np.transpose(actions,(1,0,2))

        state = state.to(torch.float32)
        actions = actions.to(torch.float32)
        cost = cost.to(torch.float32)

        state = state.to(device)
        actions = actions.to(device)
        cost = cost.to(device)

        current_cost_batch = critic_model(state, actions)
    #     # print(current_cost_batch)
    #     # print(cost_batch)
        critic_loss = criterion(current_cost_batch, cost)
        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

    
        losses.append(critic_loss.item())
    loss_list.append(np.mean(losses))
    print(np.mean(losses))

plt.figure()
plt.plot(loss_list)
plt.show()