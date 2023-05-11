import random
import torch
import torch.nn as nn
import torch.nn.functional as F

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]

        self.current_eps = self.eps_start

        self.fc1 = nn.Linear(4, 256)
        self.fc2 = nn.Linear(256, self.n_actions)

        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

        self.env = env_config["env_name"]
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
        self.fc1conv = nn.Linear(3136, 512)
        self.fc2conv = nn.Linear(512, self.n_actions)

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        if self.env == "cartpole":
            x = self.relu(self.fc1(x))
            x = self.fc2(x)
        elif self.env == "pong":
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.relu(self.conv3(x))
            x = self.flatten(x)
            x = self.relu(self.fc1conv(x))
            x = self.fc2conv(x)
        else:
            raise ValueError("Unknown environment: {}".format(self.env))

        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # Implement action selection using the Deep Q-network. This function
        # takes an observation tensor and should return a tensor of actions.
        # For example, if the state dimension is 4 and the batch size is 32,
        # the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        # Implement epsilon-greedy exploration.
        if (not exploit) and (random.random() < self.current_eps):
            actions = torch.randint(
                0, self.n_actions, (observation.shape[0], 1)).to(device)
        else:
            actions = torch.argmax(self.forward(
                observation), dim=1).to(device).unsqueeze(1)

        return actions


def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # Sample a batch from the replay memory and concatenate so that there are
    # four tensors in total: observations, actions, next observations and rewards.
    # Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    # Note that special care is needed for terminal transitions!
    obss, actions, next_obss, rewards = memory.sample(dqn.batch_size)

    # Stack tensors
    obss = torch.stack(obss).to(device)
    actions = torch.stack(actions).to(device)
    next_obss = torch.stack(next_obss).to(device)
    rewards = torch.stack(rewards).to(device)

    # Compute the current estimates of the Q-values for each state-action
    # pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    # corresponding to the chosen actions.

    # obss has shape [batch_size, 1, a_space], actions has shape [batch_size]
    # actions.view instead has shape [batch_size, 1, 1]
    # q_values = torch.gather(dqn.forward(
    #     obss), 2, actions.view(-1, 1, 1)).squeeze(2).to(device)

    q_values = torch.zeros(dqn.batch_size, device=device)
    for obs, action, i in zip(obss, actions, range(dqn.batch_size)):
        q_value = dqn.forward(obs)
        q_values[i] = q_value[0, action]

    q_values = q_values.unsqueeze(1)

    # Compute the Q-value targets. Only do this for non-terminal transitions!
    q_value_targets = torch.zeros(dqn.batch_size, device=device)
    for next_obs, reward, i in zip(next_obss, rewards, range(dqn.batch_size)):

        if torch.count_nonzero(next_obs) != 0:
            q_value_targets[i] = reward + dqn.gamma * \
                torch.max(target_dqn.forward(next_obs)).to(device)
        else:
            q_value_targets[i] = reward

    dqn.current_eps = max([dqn.eps_end, dqn.current_eps -
                          (dqn.eps_start - dqn.eps_end) / dqn.anneal_length])

    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)

    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()
