import torch
import torchvision.transforms as T
import numpy as np

from gymnasium.wrappers import LazyFrames

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(obs, env, last_obss=None):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v1']:
        return torch.tensor(obs, device=device).float()
    elif env in ['ALE/Pong-v5']:

        # Bin all color values between 0-255 in all frames to be either 0 or 1
        #binned_frames = [np.where(frame < 128, 0, 1) for frame in obs]
        #obs = LazyFrames(binned_frames)
        
        normalized_frames = [frame / 255 for frame in obs]
        obs = LazyFrames(normalized_frames)

        # Convert to a numpy array before creating a tensor (otherwise we get
        # a warning because this is supposed to be really efficient)
        obs = np.array(obs)

        return torch.tensor(obs, device=device).float()
    else:
        raise ValueError(
            'Please add necessary observation preprocessing instructions to preprocess() in utils.py.')
