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
        policy: A function that takes in a state and
        returns the next action to take.
        episodes: The total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.

    Returns:
        V: The updated value estimate.
    """
    for episode in range(episodes):
        # Generate an episode
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        episode_data = []

        for step in range(max_steps):
            # Select action based on policy
            action = policy(state)

            # Take action and observe the result
            result = env.step(action)
            next_state, reward, terminated, truncated = result[:4]

            # Store the state and reward in the episode history
            episode_data.append((state, reward))

            # If the episode is terminated or truncated, end the episode
            if terminated or truncated:
                break

            # Move to the next state
            state = next_state

        # Initialize the return (G) to 0
        G = 0
        episode_data = np.array(episode_data, dtype=int)

        # Compute the returns for each state in the episode,
        # starting from the end
        for state, reward in reversed(episode_data):
            # Calculate the return for this state
            G = reward + gamma * G

            # If this state has not been visited in this episode
            if state not in episode_data[:, 0]:
                # Update the value function V(s) using the
                # incremental update formula
                V[state] = V[state] + alpha * (G - V[state])

    return V
