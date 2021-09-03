
# from lib import wrapper
# from lib import dqn_model
import argparse
import time
import random
import collections

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import gym

from torch.utils.tensorboard import SummaryWriter



Experience = collections.namedtuple(
    'Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])


class AgentDQN:
    def __init__(self, qnet: nn.Module, buffer_capacity: int, gamma=0.99, lr=1e-4) -> None:
        self.qnet = qnet
        self.buffer_capacity = buffer_capacity
        self.gamma = gamma
        self.optimizer = optim.Adam(self.qnet.parameters(), lr=lr)
        self.buffer = collections.deque(maxlen=buffer_capacity)
        self.state = None  # current state of agent.

    def sample_from_buffer(self, batch_size: int):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states, actions, rewards, dones, next_states = zip(*[self.buffer[idx] for idx in indices])
        return np.array(states), np.array(actions), np.array(rewards, dtype=np.float32), \
            np.array(dones, dtype=np.uint8), np.array(next_states)

    def select_action(self, state: np.ndarray, exploration_rate: float = 0.1) -> int:
        if random.random() < exploration_rate: # epsilon-greedy exploration
            act_int = random.randint(0, self.action_dim-1)
        else:
            device = next(self.qnet.parameters()).device
            state = torch.as_tensor([state], dtype=torch.float32, device=device)
            act_probs = self.qnet(state)
            act_int = act_probs.argmax(dim=1).detach().cpu().item()
        return act_int

    def explore_env(self, env: gym.Env, n_steps: int = 1):
        """Explore the current with the current Q-Net by the given steps. All
        experience steps will be stored in the replay buffer.
        """
        if self.state is None:
            self.state = env.reset()
        for _ in range(n_steps):
            action = self.select_action(self.state)
            next_state, reward, is_done, _ = env.step(action=action)

            exp = Experience(self.state, action, reward, is_done, next_state)
            self.buffer.append(exp)

            if is_done:
                self.state = env.reset()

    def update_net(self, tgt_net: nn.Module, batch_size):
        s, actions, rewards, is_dones, next_s = self.sample_from_buffer(batch_size)
        s = torch.as_tensor(s, dtype=torch.float32)
        actions = torch.as_tensor(actions, dtype=torch.int64)
        next_s = torch.as_tensor(next_s, dtype=torch.float32)
        rewards = torch.as_tensor(rewards, dtype=torch.float32)
        is_dones = torch.as_tensor(is_dones, dtype=torch.bool)

        # actions are not necessary the best ones any more.
        q = self.net(s).gather(dim=1, index=actions.unsqueeze(dim=1))
        q_next = tgt_net(s).max(dim=1)[0]

        # discount q_next with gamma, then if is_done is True, q_next set to 0.
        q_next_mask = (q_next * self.gamma) * (~is_dones)
        self.optimizer.zero_grad()
        loss = F.mse_loss(q, rewards + q_next_mask.detach())

        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    env_name = 'PongNoFrameskip-v4'
    mean_reward_bound = 19.5

    gamma = 0.99
    batch_size = 32
    replay_size = 10000
    learning_rate = 1e-4
    sync_target_frames = 1000
    replay_start_size = 10000

    epsilon_decay_last_frame = 1e5
    epsilon_start = 1.0
    epsilon_final = 0.02
    n_exploration_step = 10
    log_interval = 10

    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default=env_name,
                        help="Name of the environment, default=" + env_name)
    parser.add_argument("--reward", type=float, default=mean_reward_bound,
                        help=f"Mean reward boundary for stop of training, default={mean_reward_bound:.2f}")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = wrapper.make_env(env_name)

    net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = dqn_model.DQN(env.observation_space.shape, env.action_space.n).to(device)
    writer = SummaryWriter(comment=f'_{env_name}')

    agent = AgentDQN(net, buffer_capacity=replay_start_size, gamma=gamma)

    epsilon = epsilon_start

    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()

    step_idx = 0
    while True:
        step_idx +=1
        frame_idx = step_idx * n_exploration_step
        epsilon = max(epsilon_final, epsilon_start - frame_idx / epsilon_decay_last_frame)
        agent.explore_env(env, n_steps=n_exploration_step)

        if len(agent.buffer) < replay_start_size:
            continue

        if step_idx % log_interval == 1:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts = time.time()

            mean_reward = np.mean(total_rewards[-100:])
            print((f"frame {frame_idx}, done {len(total_rewards)} games,"
                  f" mean_reward: {mean_reward:.3f}, eps: {epsilon:.2f}, speed: {speed:.2f} f/s."))
            writer.add_scalar('epsilon', epsilon, frame_idx)
            writer.add_scalar('speed', speed, frame_idx)
            writer.add_scalar('reward_100', mean_reward, frame_idx)
            writer.add_scalar('reward', reward, frame_idx)



        if frame_idx % sync_target_frames == 0: # tgt is lagged behind.
            tgt_net.load_state_dict(net.state_dict())


    writer.close()







