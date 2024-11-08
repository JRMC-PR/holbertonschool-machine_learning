#!/usr/bin/env python3
"""This module contains the function for the SARSA(λ) algorithm."""
import numpy as np


def sarsa_lambtha(env, Q, lambtha, episodes=5000,
                  max_steps=100, alpha=0.1, gamma=0.99,
                  epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    """
    Performs the SARSA(λ) algorithm to update the Q table.

    Args:
        env: The environment instance.
        Q: A numpy.ndarray of shape (s, a) containing the Q table.
        lambtha: The eligibility trace factor.
        episodes: The total number of episodes to train over.
        max_steps: The maximum number of steps per episode.
        alpha: The learning rate.
        gamma: The discount rate.
        epsilon: The initial threshold for epsilon greedy.
        min_epsilon: The minimum value that epsilon should decay to.
        epsilon_decay: The decay rate for updating epsilon between episodes.

    Returns:
        Q: The updated Q table.
    """
    def epsilon_greedy(state, Q, epsilon):
        """Selects an action using epsilon-greedy policy."""
        if np.random.rand() < epsilon:
            return env.action_space.sample()
        else:
            return np.argmax(Q[state])

    for episode in range(episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]
        action = epsilon_greedy(state, Q, epsilon)
        E = np.zeros_like(Q)  # Initialize the eligibility trace

        for step in range(max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            if isinstance(next_state, tuple):
                next_state = next_state[0]
            next_action = epsilon_greedy(next_state, Q, epsilon)

            # Calculate the TD error
            delta = reward + gamma * Q[next_state,
                                       next_action] - Q[state, action]

            # Update the eligibility trace
            E[state, action] += 1

            # Update the Q table and eligibility trace
            Q += alpha * delta * E
            E *= gamma * lambtha

            if terminated or truncated:
                break

            state = next_state
            action = next_action

        # Decay epsilon
        epsilon = max(min_epsilon, epsilon * np.exp(-epsilon_decay * episode))

    return Q
