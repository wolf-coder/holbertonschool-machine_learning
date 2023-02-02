#!/usr/bin/env python3
"""
TRAIN
"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result= False):
    """
    env: initial environment
    nb_episodes: number of episodes used for training
    alpha: the learning rate
    gamma: the discount factor
    """
    weight = np.random.rand(4,2)
    scores = []
    for ep in range(nb_episodes):
        state = env.reset()[None, :]
        grads, rewards = [], []
        sum_rew = 0
        while True:
            if show_result and not ep % 1000:
                env.render()
            action, grad = policy_gradient(state, weight)
            n_state, reward, done, _ = env.step(action)
            n_state = n_state[None, :]
            grads.append(grad)
            rewards.append(reward)
            sum_rew += reward
            state = n_state
            if done:
                break
        for i in range(len(grads)):
            summ = sum([r * (gamma ** r) for t, r in enumerate(rewards[i:])])
            weight += alpha * grads[i] * summ
        scores.append(sum_rew)
        print("EP: {}: score: {}".format(
            ep, sum_rew), end="\r", flush=False)
    return scores
