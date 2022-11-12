import numpy as np
import torch as T
from deep_q_network import DeepQNetwork, DuelingDeepQNetwork
from replay_memory import ReplayBuffer


class Agent():
    def __init__(self, gamma, epsilon, lr, n_actions, input_dims,
                mem_size, batch_size, eps_min=0.01, eps_dec=5e-7,
                replace=1000, algo=None, env_name=None, checkpoint_dir='tmp/dqn'):
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr
        self.n_actions = n_actions
        self.input_dims = input_dims
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.eps_min = eps_min
        self.eps_dec = eps_dec
        self.replace_target_cnt = replace
        self.algo = algo
        self.env_name = env_name
        self.checkpoint_dir = checkpoint_dir
        self.action_space = [i for i in range(self.n_actions)]
        self.learn_step_counter = 0 # Count the number of times we called the learn-function

        self.memory = ReplayBuffer(self.mem_size, self.input_dims, self.n_actions)

    def choose_action(self, observation):
        raise NotImplementedError

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.store_transition(state, action, reward, next_state, done)

    def sample_memory(self):
        state, action, reward, new_state, done = \
                self.memory.sample_buffer(self.batch_size)

        states = T.tensor(state).to(self.q_eval.device)
        actions = T.tensor(action).to(self.q_eval.device)
        rewards = T.tensor(reward).to(self.q_eval.device)
        new_states = T.tensor(new_state).to(self.q_eval.device)
        dones = T.tensor(done).to(self.q_eval.device)

        return states, actions, rewards, new_states, dones

    def replace_target_network(self):
        if self.learn_step_counter % self.replace_target_cnt == 0:
            self.q_next.load_state_dict(self.q_eval.state_dict())

    def decrement_epsilon(self):
        self.epsilon = self.epsilon - self.eps_dec \
            if self.epsilon > self.eps_min else self.eps_min

    def save_models(self):
        self.q_eval.save_checkpoint()
        self.q_next.save_checkpoint()

    def load_models(self):
        self.q_eval.load_checkpoint()
        self.q_next.load_checkpoint()

    def learn(self):
        raise NotImplementedError

class DQNAgent(Agent):
    # A little trick to not repeat the aguments of the parent class
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                input_dims=self.input_dims,
                                name=self.env_name+'_'+self.algo+'_q_eval',
                                checkpoint_dir=self.checkpoint_dir)
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                input_dims=self.input_dims,
                                name=self.env_name+'_'+self.algo+'_q_next',
                                checkpoint_dir=self.checkpoint_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, new_states, dones = self.sample_memory()

        # Incorporate the batch_size and correct indices
        indices = np.arange(self.batch_size)
        q_pred = self.q_eval.forward(states)[indices, actions]
        # [0] because the max function returns a named tuple
        q_next = self.q_next.forward(new_states).max(dim=1)[0]

        # Everywhere where done is true, we set the q_next to 0
        q_next[dones] = 0.0
        # if dones.any() == 1:
        #     print(q_next, dones)
        q_target = rewards + self.gamma * q_next

        # Calculate the loss
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

class DDQNAgent(Agent):
    def __init__(self, *args, **kvargs):
        super().__init__(*args, **kvargs)

        self.q_eval = DeepQNetwork(self.lr, self.n_actions,
                                input_dims=self.input_dims,
                                name=self.env_name+'_'+self.algo+'_q_eval',
                                checkpoint_dir=self.checkpoint_dir)
        self.q_next = DeepQNetwork(self.lr, self.n_actions,
                                input_dims=self.input_dims,
                                name=self.env_name+'_'+self.algo+'_q_next',
                                checkpoint_dir=self.checkpoint_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            actions = self.q_eval.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        # zero out the gradients
        self.q_eval.optimizer.zero_grad()

        # replace if appropriate
        self.replace_target_network()

        states, actions, rewards, next_states, dones = self.sample_memory()

        indices = np.arange(self.batch_size)

        #################################
        # Crucial part for DDQN
        # Funny that it only takes so few lines of code
        #################################
        q_pred = self.q_eval.forward(states)[indices, actions]
        q_next = self.q_next.forward(next_states)
        q_eval = self.q_eval.forward(next_states)

        # maximum actions for the next states according to the q_eval network
        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0

        q_target = rewards + self.gamma * q_next[indices, max_actions]
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()

        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

class DuelingDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                input_dims=self.input_dims,
                                name=self.env_name+'_' + self.algo + '_q_eval',
                                checkpoint_dir=self.checkpoint_dir)

        # We will never perform gradient descent/backprpagation on the q_next network
        # q_next = target network
        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                input_dims=self.input_dims,
                                name=self.env_name+'_' + self.algo + '_q_next',
                                checkpoint_dir=self.checkpoint_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            # Dueling DQN adjustments
            value, advantage = self.q_eval.forward(state)   
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, new_states, dones = self.sample_memory()

        # Incorporate the batch_size and correct indices
        indices = np.arange(self.batch_size)
        
        # Dueling DQN adjustments
        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(new_states) # taget network

        # dim=1 is the action dimension
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]
        # We want to calculate the max on the action dimension and take the 0th element
        # just as we did in regular q-learning. We want to find the maximal actions for
        # the resulting states according to the target network. We take the 0th element
        # because the max function returns a tuple with the 0th postition are the values 
        # and the 1st position are the indices
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True))).max(dim=1)[0]

        q_next[dones] = 0.0
        q_target = rewards + self.gamma * q_next

        # Calculate the loss
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()

class DuelingDDQNAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.q_eval = DuelingDeepQNetwork(self.lr, self.n_actions,
                                input_dims=self.input_dims,
                                name=self.env_name+'_' + self.algo + '_q_eval',
                                checkpoint_dir=self.checkpoint_dir)

        # We will never perform gradient descent/backprpagation on the q_next network
        # q_next = target network
        self.q_next = DuelingDeepQNetwork(self.lr, self.n_actions,
                                input_dims=self.input_dims,
                                name=self.env_name+'_' + self.algo + '_q_next',
                                checkpoint_dir=self.checkpoint_dir)

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation], dtype=T.float).to(self.q_eval.device)
            # Dueling DQN adjustments
            value, advantage = self.q_eval.forward(state)   
            action = T.argmax(advantage).item()
        else:
            action = np.random.choice(self.action_space)
        return action

    def learn(self):
        '''
        The learn function is the only part that has to be changed
        for Dueling DDQN
        '''
        if self.memory.mem_cntr < self.batch_size:
            return
        
        self.q_eval.optimizer.zero_grad()

        self.replace_target_network()

        states, actions, rewards, new_states, dones = self.sample_memory()

        # Incorporate the batch_size and correct indices
        indices = np.arange(self.batch_size)
        
        # Dueling DQN adjustments
        V_s, A_s = self.q_eval.forward(states)
        V_s_, A_s_ = self.q_next.forward(new_states) # taget network

        # Dueling DDQN adjustments
        V_s_eval, A_s_eval = self.q_eval.forward(new_states)

        # dim=1 is the action dimension
        q_pred = T.add(V_s, (A_s - A_s.mean(dim=1, keepdim=True)))[indices, actions]

        # Dueling DDQN adjustments: get rid of the max function
        q_next = T.add(V_s_, (A_s_ - A_s_.mean(dim=1, keepdim=True)))

        # Dueling DDQN adjustments
        q_eval = T.add(V_s_eval, (A_s_eval - A_s_eval.mean(dim=1, keepdim=True)))
        max_actions = T.argmax(q_eval, dim=1)

        q_next[dones] = 0.0
        # Dueling DDQN adjustments
        q_target = rewards + self.gamma * q_next[indices, max_actions]

        # Calculate the loss
        loss = self.q_eval.loss(q_target, q_pred).to(self.q_eval.device)
        loss.backward()
        self.q_eval.optimizer.step()
        self.learn_step_counter += 1

        self.decrement_epsilon()
