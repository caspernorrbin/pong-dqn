import argparse

import gymnasium as gym
import torch

import config
from utils import preprocess

from gymnasium.wrappers import AtariPreprocessing, FrameStack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v1', 'ALE/Pong-v5'], default='CartPole-v1')
parser.add_argument('--path', type=str, help='Path to stored DQN model.')
parser.add_argument('--n_eval_episodes', type=int, default=1, help='Number of evaluation episodes.', nargs='?')
parser.add_argument('--render', dest='render', action='store_true', help='Render the environment.')
parser.add_argument('--save_video', dest='save_video', action='store_true', help='Save the episodes as video.')
parser.set_defaults(render=False)
parser.set_defaults(save_video=False)

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v1': config.CartPole,
    'ALE/Pong-v5': config.Pong
}


def evaluate_policy(dqn, env, env_config, args, n_episodes, render=False, verbose=False):
    """Runs {n_episodes} episodes to evaluate current policy."""
    total_return = 0
    for i in range(n_episodes):
        obs, info = env.reset()
        obs = preprocess(obs, env=args.env).unsqueeze(0)

        terminated = False
        episode_return = 0

        while not terminated:
            if render:
                env.render()

            action = dqn.act(obs, exploit=True).item()
            
            offset = 0
            if env_config['env_name'] == "pong":
                # 0 -> 2, up
                # 1 -> 0, noop
                # 2 -> 3, down
                if env_config["n_actions"] == 2:
                    offset = 2
                elif action == 0:
                    offset = 2
                elif action == 1:
                    offset = -1
                elif action == 2:
                    offset = 1
            
            obs, reward, terminated, truncated, info = env.step(action + offset)
            obs = preprocess(obs, env=args.env).unsqueeze(0)

            episode_return += reward
        
        total_return += episode_return
        
        if verbose:
            print(f'Finished episode {i+1} with a total return of {episode_return}')

    
    return total_return / n_episodes

def main():
    args = parser.parse_args()

    # Initialize environment and config
    env = gym.make(args.env)
    env_config = ENV_CONFIGS[args.env]

    if args.save_video:
        env = gym.make(args.env, render_mode='rgb_array')
        env = gym.wrappers.RecordVideo(env, './video/', episode_trigger=lambda episode_id: True)
        
    if args.env in ["ALE/Pong-v5"]:
        env = AtariPreprocessing(
            env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)
        env = FrameStack(env, 4)

    # Load model from provided path.
    dqn = torch.load(args.path, map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    dqn.eval()

    mean_return = evaluate_policy(dqn, env, env_config, args, args.n_eval_episodes, render=args.render and not args.save_video, verbose=True)
    print(f'The policy got a mean return of {mean_return} over {args.n_eval_episodes} episodes.')

    env.close()
    

if __name__ == '__main__':
    main()
    