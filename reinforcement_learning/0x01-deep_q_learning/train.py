#!/usr/bin/env python3
"""
Deep Q Learning on Atari's Breakout v1
"""
import gym
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from rl.memory import SequentialMemory
from rl.agents import DQNAgent
from rl.policy import LinearAnnealedPolicy, EpsGreedyQPolicy


def create_q_model(h, w, ch, actions):
    """
    Network defined by the Deepmind paper
    """
    model = Sequential()
    model.add(
        layers.Convolution2D(
            32, (8, 8), strides=(
                4, 4), activation='relu', input_shape=(
                3, h, w, ch)))
    model.add(
        layers.Convolution2D(
            64, (4, 4), strides=(
                2, 2), activation='relu'))
    model.add(layers.Convolution2D(64, (3, 3), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(actions, activation='linear'))
    return model


def create_agent(model, actions):
    """build agent method"""
    policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps',
                                  value_max=1.,
                                  value_test=.2,
                                  value_min=.1,
                                  nb_steps=10000)
    memory = SequentialMemory(limit=1000000, window_length=3)
    dq_agent = DQNAgent(model=model, memory=memory, policy=policy,
                        enable_dueling_network=True,
                        dueling_type='avg',
                        nb_actions=actions,
                        nb_steps_warmup=1000)
    return dq_agent


def train(env):
    env.reset()
    height, width, channels = env.observation_space.shape
    actions = env.action_space.n
    conv_model = create_q_model(h=height, w=width,
                                ch=channels, actions=actions)
    print(conv_model.summary())
    dqn_agent = create_agent(conv_model, actions)
    dqn_agent.compile(optimizer=Adam(lr=1e-3), metrics=['mae'])
    dqn_agent.fit(env, nb_steps=10000, visualize=False, verbose=2)
    dqn_agent.save_weights('policy.h5', overwrite=True)


env = gym.make('Breakout-v0', render_mode='human')
train(env)
