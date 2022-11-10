import gym
import numpy as np
import matplotlib.pyplot as plt

# The API changed a bit here: 
# https://stackoverflow.com/questions/73195438/openai-gyms-env-step-what-are-the-values
# new_step_api=True would return a 5-tuple with an additional truncated boolean flag
env = gym.make('FrozenLake-v1')

n_games = 1000
win_percentages = []
scores = []

for i in range(n_games):
    terminated = False
    obs, _ = env.reset()
    score = 0
    while not terminated:
        action = env.action_space.sample() # random action
        next_state, reward, terminated, truncated , info = env.step(action)
        score += reward
    scores.append(score)

    if i % 10 == 0:
        average = np.mean(scores[-10:])
        win_percentages.append(average)

plt.plot(win_percentages)
plt.show()