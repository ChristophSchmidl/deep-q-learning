import numpy as np
from dqn_agent import DQNAgent
from utils import plot_learning_curve, make_env


if __name__ == '__main__':
    env = make_env('PongNoFrameskip-v4')
    best_score = -np.inf
    load_checkpoint = False
    n_games = 500
    agent = DQNAgent(gamma=0.99, epsilon=1.0, lr=0.0001,
                    input_dims=(env.observation_space.shape),
                    n_actions=env.action_space.n, mem_size=15000,
                    eps_min=0.1, batch_size=32, replace=1000,
                    eps_dec=1e-5, checkpoint_dir='models/', algo='DQNAgent',
                    env_name='PongNoFrameskip-v4')

    if load_checkpoint:
        agent.load_models()

    filename = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' + \
                str(n_games) + 'games'
    figure_file = 'plots/' + filename + '.png'
    
    n_steps = 0
    scores, epsilon_history, steps_array = [], [], []

    for i in range(n_games):
        terminated = False
        score = 0
        obs = env.reset()
        
        while not terminated:
            action = agent.choose_action(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            score += reward

            if not load_checkpoint:
                agent.store_transition(obs, action, reward, next_obs, int(terminated))
                agent.learn()
            
            obs = next_obs
            n_steps += 1
        scores.append(score)
        steps_array.append(n_steps)

        avg_score = np.mean(scores[-100:])
        print(f"Episode: {i}, Score: {score:.2f}, Average score: {avg_score:.2f}, Epsilon: {agent.epsilon:.2f}, Steps: {n_steps}")

        if avg_score > best_score:
            if not load_checkpoint:
                agent.save_models()
            best_score = avg_score

        epsilon_history.append(agent.epsilon)
    
    plot_learning_curve(steps_array, scores, epsilon_history, figure_file)
