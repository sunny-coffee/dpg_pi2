import numpy as np 
import torch

class Buffer:
    def __init__(self, num_states, num_actions, n_steps, buffer_capacity=1000, batch_size=64):
        # Number of "experiences" to store at max
        self.buffer_capacity = buffer_capacity
        # Num of tuples to train on.
        self.batch_size = batch_size

        # Its tells us num of times record() was called.
        self.buffer_counter = 0

        # Instead of list of tuples as the exp.replay concept go
        # We use different np.arrays for each tuple element
        self.state_buffer = np.zeros((self.buffer_capacity, num_states))
        self.actions_buffer = np.zeros((self.buffer_capacity, n_steps, num_actions))
        self.cost_buffer = np.zeros((self.buffer_capacity, 1))
        # self.next_state_buffer = np.zeros((self.buffer_capacity, num_states))

    # Takes (s,actions,r) obervation tuple as input
    def record(self, obs_tuple):
        # Set index to zero if buffer_capacity is exceeded,
        # replacing old records
        record_size = obs_tuple[0].shape[0]
        for i in range(record_size):
            index = self.buffer_counter % self.buffer_capacity
            self.state_buffer[index] = obs_tuple[0][i].detach().numpy()
            self.actions_buffer[index] = obs_tuple[1][:,i,:].detach().numpy()
            self.cost_buffer[index] = obs_tuple[2][i]
            # self.next_state_buffer[index] = obs_tuple[3]
            self.buffer_counter += 1

    def sample(self):
        # Get sampling range
        record_range = min(self.buffer_counter, self.buffer_capacity)
        # Randomly sample indices
        batch_indices = np.random.choice(record_range, self.batch_size)

        # Convert to tensors
        state_batch = torch.from_numpy(np.float32(self.state_buffer[batch_indices]))
        actions_batch = torch.from_numpy(np.float32(np.transpose(self.actions_buffer[batch_indices], (1,0,2))))
        cost_batch = torch.from_numpy(np.float32(self.cost_buffer[batch_indices]))

        return state_batch, actions_batch, cost_batch

    def save_data(self):
        np.save('./data/state', self.state_buffer[0:self.buffer_counter])
        np.save('./data/actions', self.actions_buffer[0:self.buffer_counter])
        np.save('./data/cost', self.cost_buffer[0:self.buffer_counter])
        print(self.buffer_counter)
    