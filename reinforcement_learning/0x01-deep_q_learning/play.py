#!/usr/bin/env python3
"""
Deep Q Learning on Atari's Breakout v1
"""
from rl.memory import SequentialMemory
from rl.policy import GreedyQPolicy
import gym
from rl.agents.dqn import DQNAgent
import keras as K
from tensorflow.keras.optimizers import Adam
train = __import__('train').train

env = gym.make('Breakout-v0')
state = env.reset()
model = K.models.load_model('policy.h5')
DQN = DQNAgent(
    model=model,
    nb_actions=env.action_space.n,
    memory=SequentialMemory(limit=100000, window_length=3),
    policy=GreedyQPolicy())
DQN.compile(optimizer=Adam(
    lr=1e-3,
    clipnorm=1.0), metrics=['mae'])
DQN.test(
    env,
    nb_episodes=10,
    visualize=True,
)
