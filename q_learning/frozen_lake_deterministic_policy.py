import gym
import numpy as np
import matplotlib.pyplot as plt


# LEFT=0, DOWN=1, RIGHT=2, UP=3
# S=Start,G=Goal,F=Frozen,H=Hole
# SFFF
# FHFH
# FFFH
# HFFG

# Hard-coded policy: state->action
policy = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1, 8: 2, 9: 1, 10: 1, 13: 2, 14: 2}

env = gym.make('FrozenLake-v1')

n_games = 1000
win_percentages = []
scores = []

for i in range(n_games):
    terminated = False
    obs, _ = env.reset()
    score = 0
    while not terminated:
        action = policy[obs]
        next_state, reward, terminated, truncated, info = env.step(action)
        score += reward
    scores.append(score)

    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_percentages.append(average)
    
plt.plot(win_percentages)
plt.show()