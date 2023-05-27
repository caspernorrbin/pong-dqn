import argparse

import gymnasium as gym
import torch

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

from gymnasium.wrappers import AtariPreprocessing, FrameStack

import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument(
    '--env', choices=['CartPole-v1', 'ALE/Pong-v5'], default='CartPole-v1')
parser.add_argument('--evaluate_freq', type=int, default=25,
                    help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5,
                    help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole,
    'ALE/Pong-v5': config.Pong
}

def main():
    args = parser.parse_args()

    # Initialize environment and config.
    env = gym.make(args.env)

    if args.env in ["ALE/Pong-v5"]:
        env = AtariPreprocessing(
            env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
        env = FrameStack(env, 4)

    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # Create and initialize target Q-network.
    dqn_target = DQN(env_config=env_config).to(device)

    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")

    optimize_count = 0
    update_target_count = 0
    
    rewards = []

    for episode in range(env_config['n_episodes']):
        terminated = False
        obs, info = env.reset()

        obs = preprocess(obs, env=args.env).unsqueeze(0)

        while not terminated:
            # Get action from DQN.
            # print(dqn.act(obs))
            action = dqn.act(obs).item()
            curr_obs = obs
            
            offset = 0
            if env_config['env_name'] == "pong":
                # 0 -> 2
                # 1 -> 0
                # 2 -> 3
                if env_config["n_actions"] == 2:
                    offset = 2
                elif action == 0:
                    offset = 2
                elif action == 1:
                    offset = -1
                elif action == 2:
                    offset = 1

            # print(env.action_space)
            # Act in the true environment.
            obs, reward, terminated, truncated, info = env.step(action + offset)

            # Preprocess incoming observation.
            if not terminated:
                obs = preprocess(obs, env=args.env,
                                 last_obss=curr_obs).unsqueeze(0)
            else:
                obs = torch.zeros(obs.shape, device=device).unsqueeze(0)

            # Add the transition to the replay memory. Remember to convert
            # everything to PyTorch tensors!
            memory.push(curr_obs, torch.tensor(action, device=device),
                        obs, torch.tensor(reward, device=device))

            # Run DQN.optimize() every env_config["train_frequency"] steps.
            if optimize_count % env_config["train_frequency"] == 0:
                optimize(dqn, dqn_target, memory, optimizer)
            optimize_count += 1

            # Update the target network every env_config["target_update_frequency"] steps.
            if update_target_count % env_config["target_update_frequency"] == 0:
                dqn_target.load_state_dict(dqn.state_dict())
            update_target_count += 1

        # Evaluate the current agent.
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(
                dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            print(
                f'Episode {episode+1}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')
                
            rewards.append(mean_return)

    y = [1 + args.evaluate_freq * i for i in range(len(rewards))]
    
    plt.plot(y, rewards)
    plt.xlabel("Episode")
    plt.ylabel("Mean Return")
    plt.title("Mean Return vs Episode")
    plt.savefig("mean_return.png")
    
    # Close environment after training is completed.
    env.close()

if __name__ == '__main__':
    main()
    