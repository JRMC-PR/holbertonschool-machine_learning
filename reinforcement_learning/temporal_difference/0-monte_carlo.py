#!/usr/bin/env python3
"""This module contains the function for the Monte Carlo algorithm."""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000,
                max_steps=100, alpha=0.1, gamma=0.99):
    """
    Performs the Monte Carlo algorithm to estimate the value function.

    Args:
        env: The environment instance.
        V: A numpy.ndarray of shape (s,) containing the value estimate.
        policy: A function that takes in a state and returns the next
        action to take.
        episodes: The total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.

    Returns:
        V: The updated value estimate.
    """
    for episode in range(episodes):
        # Generate an episode
        state = env.reset()  # Unpack the tuple returned by env.reset()
        if isinstance(state, tuple):
            state = state[0]
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            result = env.step(action)
            next_state, reward, done = result[:3]
            episode_data.append((state, action, reward))
            if done:
                break
            state = next_state

        # Calculate returns and update value function
        G = 0
        for state, action, reward in reversed(episode_data):
            G = reward + gamma * G
            V[state] = V[state] + alpha * (G - V[state])

    return V
