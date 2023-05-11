import torch
import torchvision.transforms as T
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def preprocess(obs, env, last_obss=None):
    """Performs necessary observation preprocessing."""
    if env in ['CartPole-v1']:
        return torch.tensor(obs, device=device).float()
    elif env in ['ALE/Pong-v5']:
        return torch.tensor(obs, device=device).float()
        print(obs)
        exit()

        processed = np.where(obs[0] < 128, 0, 1)

        if last_obss is None:
            obs = np.stack([obs, obs, obs, obs], axis=0)
        else:
            obs.append(processed)
            obs.pop(0)

        im = cv2.imwrite("test.png", obs[0])
        print(obs[0][5])

        print(np.where(obs[0] < 128, 0, 1))
        obs[0] = obs[0]._force()
        cv2.imwrite("test2.png", obs[0])

        print(obs[0][5])

        exit()
        return torch.tensor(obs, device=device).float()
    else:
        raise ValueError(
            'Please add necessary observation preprocessing instructions to preprocess() in utils.py.')
