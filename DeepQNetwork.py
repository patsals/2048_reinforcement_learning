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

        nn.init.xavier_normal_(self.fc1.weight)
        nn.init.xavier_normal_(self.fc2.weight)
        nn.init.xavier_normal_(self.fc3.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = F.softmax(self.fc3(x))

        return actions
    

class Agent():
    def __init__(self, gamma, epsilon, lr, input_dims, batch_size, n_actions, 
                 max_mem_size=100_000, eps_end=0.01, eps_dec=5e-4, device='cpu'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.input_dims = input_dims 
        self.batch_size = batch_size 
        self.n_actions = n_actions
        self.mem_size = max_mem_size 
        self.eps_end = eps_end
        self.eps_dec = eps_dec
        self.device = device

        self.action_space = [i for i in range(n_actions)]
        self.mem_counter = 0 
        self.Q_eval = DeepQNetwork(self.lr, n_actions=n_actions, input_dims=input_dims,
                                  fc1_dims=256, fc2_dims=256)
        
        self.Q_eval.to(device)

        self.state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros((self.mem_size, *input_dims), dtype=np.float32)
        self.action_memory = np.zeros((self.mem_size, n_actions), dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_new, done):
        index = self.mem_counter % self.mem_size
        self.state_memory[index] = state 
        self.new_state_memory[index] = state_new
        self.reward_memory[index] = reward 
        self.action_memory[index][action] = 1
        self.terminal_memory[index] = done

        self.mem_counter += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = torch.tensor([observation]).to(self.device)
            with torch.no_grad(): actions = self.Q_eval.forward(state)
            action = torch.arange(actions.size(dim=-1)).to(self.device)[actions.multinomial(num_samples=1).item()].item()
        else:
            action = np.random.choice(self.action_space)

        return action
    

    def learn(self, state, action, reward):
        # if self.mem_counter < self.batch_size:
        #     return
        
        self.Q_eval.requires_grad_(True)
        self.Q_eval.optimizer.zero_grad()

        # max_mem = min(self.mem_counter, self.mem_size)
        # batch = np.arange(num_moves)

        # batch_index = np.arange(self.batch_size, dtype=np.int32)
        # state_batch = torch.tensor(self.state_memory[batch]).to(self.device)
        # new_state_batch = torch.tensor(self.new_state_memory[batch]).to(self.device)
        # reward_batch = torch.tensor(self.reward_memory[batch]).to(self.device)
        # terminal_batch = torch.tensor(self.terminal_memory[batch]).to(self.device)
        
        # action_batch = torch.tensor(self.action_memory[batch], dtype=torch.float32).to(self.device)
        state_t = torch.tensor(state).to(self.device)
        q_eval = self.Q_eval.forward(state_t)

        #with torch.no_grad(): print(f'weights: {self.Q_eval.fc3.weight.cpu().numpy()}, out: {q_eval.cpu().numpy()}, reward: {reward}')

        action_vec = torch.zeros(self.n_actions).to(self.device)
        action_vec[action] = 1

        loss = reward*self.Q_eval.loss(q_eval, action_vec)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.Q_eval.requires_grad_(False)

        self.mem_counter = 0

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_end else self.eps_end

