import argparse
import gym
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='CartPole-v0')
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole
}

def plot_train(x, xlabel, ylabel, title, y, episodes, z=1):
    plt.plot(range(1, episodes+1, z), x)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    #plt.savefig(f'figures/{y}')
    plt.show()


if __name__ == '__main__':
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # TODO: Create and initialize target Q-network.
    target_dqn = DQN(env_config=env_config).to(device)
    target_dqn.load_state_dict(dqn.state_dict())
    target_dqn.eval()
    
    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    num_step = 0
    mean_rewards, rewards, full_rewards, mean_losses = [], [], [], []

    for episode in range(env_config['n_episodes']):
        done = False

        obs = preprocess(env.reset(), env=args.env).unsqueeze(0)
        total_reward , total_loss = 0, 0
        
        while not done:
            # TODO: Get action from DQN.
            action = dqn.act(obs)

            # Act in the true environment.
            next_obs, reward, done, info = env.step(action.item())
            total_reward += reward
            reward = torch.tensor([reward], device=device)
            
            # Preprocess incoming observation.
            if not done:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0)

            else:
                next_obs = None
            memory.push(obs, action, next_obs, reward)
            
            obs = next_obs
            num_step += 1

            if num_step % env_config["train_frequency"] == 0:
                l = optimize(dqn, target_dqn, memory, optimizer)
                if l != None:
                    total_loss += l
            
            if episode % env_config["target_update_frequency"] == 0:
                target_dqn.load_state_dict(dqn.state_dict())

        mean_losses.append(total_loss/(num_step/env_config["train_frequency"]))
        full_rewards.append(total_reward)    
        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return, total_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            mean_rewards.append(mean_return)
            rewards.append(total_return)
            print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                #torch.save(dqn, f'./models/{args.env}_best.pt')

    plot_train(mean_rewards, 'Episodes', 'Mean Reward', 'Evaluation of Mean Reward Every 25 Episodes', 'mean_reward.jpg', env_config['n_episodes'], args.evaluate_freq)
    plot_train(full_rewards, 'Episodes', 'Total Reward', 'Evaluation of Total Reward for All Episodes', 'full_reward.jpg', env_config['n_episodes'])
    plot_train(mean_losses, 'Episodes', 'Mean Loss', 'Evaluation of Mean Loss for All Episodes', 'mean_loss.jpg', env_config['n_episodes'])  

    # Close environment after training is completed.
    env.close()
