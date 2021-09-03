import argparse

import torch
import gym
from torch.utils.tensorboard import SummaryWriter
from agents import AgentDQN
from models import MLP


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--env", default='CartPole-v0',
                        help="Name of the environment")
    parser.add_argument("--stop_reward", type=float, default=19,
                        help=f"Mean reward boundary for stop of training")
    args = parser.parse_args()
    device = torch.device("cuda" if args.cuda else "cpu")

    env = gym.make(args.env)

    net = MLP(env.observation_space.shape[0], (16,), env.action_space.n)

    agent = AgentDQN(net, buffer_capacity=1000)

    step_idx = 0
    while True:
        step_idx += 1

