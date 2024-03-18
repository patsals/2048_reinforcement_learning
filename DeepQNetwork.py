# https://www.youtube.com/watch?v=wc-FxNENg9U
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQNetwork(nn.Module):
    def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = self.fc3(x)

        return actions


class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions,
                 max_mem_size=100_000, eps_min=0.01, eps_max=0.20, eps_chg=5e-4, device='cpu'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.mem_size = max_mem_size
        self.eps_min = eps_min
        self.eps_max = eps_max
        self.eps_chg = eps_chg
        self.device = device

        self.action_space = [i for i in range(n_actions)]
        self.mem_counter = 0
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                                   fc1_dims=256, fc2_dims=256)

        self.Q_eval.to(device)

        self.state_memory = np.zeros(
            (self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros(
            (self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_new, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_new
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.device)
            with torch.no_grad():
                actions = self.Q_eval.forward(state)
            amin = torch.min(actions).item()
            amax = torch.max(actions).item()
            p = (actions - amin)/(amax - amin)
            action = torch.arange(actions.size(
                dim=-1)).to(self.device)[p.multinomial(num_samples=1).item()].item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.mem_counter < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_counter, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        state_batch = torch.tensor(self.state_memory[batch]).to(self.device)
        new_state_batch = torch.tensor(
            self.new_state_memory[batch]).to(self.device)
        reward_batch = torch.tensor(self.reward_memory[batch]).to(self.device)
        terminal_batch = torch.tensor(
            self.terminal_memory[batch]).to(self.device)

        action_batch = self.action_memory[batch]
        q_eval = self.Q_eval.forward(state_batch)[batch_index, action_batch]
        q_next = self.Q_eval.forward(new_state_batch)
        q_next = torch.clone(q_next)
        q_next[terminal_batch] = 0.0
        q_target = reward_batch + self.gamma * torch.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval).to(self.device)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = self.epsilon - \
            self.eps_chg if self.epsilon > self.eps_min else self.eps_min

    def update_epsilon(self, prev_score, curr_score):

        if curr_score > prev_score:
            self.epsilon = self.epsilon - self.eps_chg if self.epsilon > self.eps_min else self.eps_min
        else:
            self.epsilon = self.epsilon + self.eps_chg if self.epsilon < self.eps_max else self.eps_max
