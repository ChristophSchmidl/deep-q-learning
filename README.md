# Deep Q-learning
Based on the github repo "Deep-Q-Learning-Paper-To-Code" by Phil Tabor. However, this repo works with Python 3.9 and newer OpenAI gym environments where the ``step`` function returns a tuple of ``observation,reward,terminated,truncated,info`` and the ``reset`` function returns a tuple of ``observation,info``.

## Installation

- ``python -m venv venv ``
- ``source venv/bin/activate`` (if you are using Linux)
- ``pip install --upgrade pip`` (optional)
- ``pip install -r requirements.txt``

If you run into compatibility issues between Pytorch and CUDA, in my case I could resolve that by installing a specific Pytorch version for CUDA 11:

- ``pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html``

## Usage

**Training**

- ``python main.py -n_games 1 -lr 0.0001 && main.py -n_games 2 -lr 0.001``

**Evaluation**

- ``python main.py -n_games 10 -eps 0 -eps_min 0 -load_checkpoint True``

## Notes

The following folders have been refactored

- ``DQN``
- ``DoubleDQN``
- ``DuelingDQN``
- ``DuelingDDQN``

into the following top-level files with better structure and inheritance:

- ``agents.py`` Here you can find the different agents such as DQNAgent, DDQNAgent, DuelingDQNAgent and DuelingDDQNAgent
- ``main.py`` Here you can find the argparser and the training loop
- ``deep_q_network.py`` Here you can find the DeepQNetwork (with CNN preprocessing) and the DuelingDeepQNetwork
- ``replay_memory.py``
- ``utils.py``

# Papers

- DQN: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)
- DQN: [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236)
- Double DQN: [Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461)
- Dueling DQN: [Dueling Network Architectures for Deep Reinforcement Learning](https://arxiv.org/abs/1511.06581)

**Other extensions**

- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783)
- [A Distributional Perspective on Reinforcement Learning](https://arxiv.org/abs/1707.06887)
- [Noisy Networks for Exploration](https://arxiv.org/abs/1706.10295)
 