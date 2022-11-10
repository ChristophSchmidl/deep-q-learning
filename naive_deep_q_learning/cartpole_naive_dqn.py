import gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch as T
from util import plot_learning_curve


class LinearDeepQNetwork(nn.Module):
    def __init__(self, lr, n_actions, input_dims):
        super().__init__()

        self.fc1 = nn.Linear(*input_dims, 128)
        self.fc2 = nn.Linear(128, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        layer1 = F.relu(self.fc1(state))
        actions = self.fc2(layer1)

        return actions


class Agent():
    def __init__(self, input_dims, n_actions, lr, gamma=0.99,
                epsilon=1.0, eps_dec=1e-5, eps_min=0.01):
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = eps_dec
        self.eps_min = eps_min
        self.action_space = [i for i in range(self.n_actions)]

        self.Q = LinearDeepQNetwork(self.lr, self.n_actions, input_dims=self.input_dims)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.Q.device)
            actions = self.Q.forward(state) # actions is a tensor
            action = T.argmax(actions).item() # action is an int
        else:
            action = np.random.choice(self.action_space)

        return action

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > self.eps_min else \
                       self.eps_min

    def learn(self, state, action, reward, next_state):
        self.Q.optimizer.zero_grad()

        states = T.tensor(state, dtype=T.float).to(self.Q.device)
        actions = T.tensor(action).to(self.Q.device)
        rewards = T.tensor(reward).to(self.Q.device)
        next_states = T.tensor(next_state, dtype=T.float).to(self.Q.device)

        # Prediction values for the current state of the env
        q_pred = self.Q.forward(states)[actions]

        q_next = self.Q.forward(next_states).max()

        q_target = reward + self.gamma * q_next

        loss = self.Q.loss(q_target, q_pred).to(self.Q.device)
        loss.backward()
        self.Q.optimizer.step()
        self.decrement_epsilon()


if __name__ == '__main__':
    env = gym.make('CartPole-v1')
    n_games = 10000
    scores = []
    epsilon_history = []

    agent = Agent(lr=0.0001, input_dims=env.observation_space.shape,
                 n_actions=env.action_space.n)

    for i in range(n_games):
        score = 0
        terminated = False
        obs, _ = env.reset()

        while not terminated:
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            score += reward
            agent.learn(obs, action, reward, next_obs)
            obs = next_obs
        scores.append(score)
        epsilon_history.append(agent.epsilon)

        if i % 100 == 0:
            avg_score = np.mean(scores[-100:])
            print(f'episode {i}, score: {score:.1f}, average score: {avg_score:.1f}, epsilon: {agent.epsilon:.2f}')

    filename = 'cartpole_naive_dqn.png'
    x = [i+1 for i in range(n_games)] # x-axis
    plot_learning_curve(x, scores, epsilon_history, filename)