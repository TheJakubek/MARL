"""Exploration strategies for MARL.

Two variants share the same epsilon schedule and 'who explores' decision;
they differ in how the exploring agents pick their random actions.
"""

import numpy as np
from scipy.stats import norm


class IndependentEpsilonGreedy:
    """Each agent independently picks a uniform-random action with prob epsilon."""

    def __init__(self, n_agents: int, n_actions: int, rng=None):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.rng = rng or np.random.default_rng()

    def select(self, q_values_list, obs_list, epsilon):
        actions = []
        for i in range(self.n_agents):
            if self.rng.random() < epsilon:
                actions.append(int(self.rng.integers(self.n_actions)))
            else:
                actions.append(int(np.argmax(q_values_list[i])))
        return actions


class CorrelatedEpsilonGreedy:
    """Random actions are sampled from a Gaussian copula whose correlation
    matrix is the cosine-similarity Gram matrix of agents' observations."""

    def __init__(self, n_agents: int, n_actions: int, rng=None, jitter: float = 1e-4):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.rng = rng or np.random.default_rng()
        self.jitter = jitter

    def _correlation_matrix(self, obs_list):
        # Stack obs into rows, L2-normalize, take outer product -> Gram matrix.
        X = np.stack(obs_list, axis=0)  # (n_agents, obs_dim)
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
        Xn = X / norms
        Sigma = Xn @ Xn.T  # (n_agents, n_agents), PSD by construction
        # Add tiny jitter for numerical stability of Cholesky.
        Sigma = Sigma + self.jitter * np.eye(self.n_agents)
        return Sigma

    def _sample_correlated_uniforms(self, Sigma):
        L = np.linalg.cholesky(Sigma)
        z_iid = self.rng.standard_normal(self.n_agents)
        z = L @ z_iid  # correlated standard normals
        u = norm.cdf(z)  # marginal Uniform(0,1), still correlated
        return u

    def select(self, q_values_list, obs_list, epsilon):
        # Step 1: decide who explores (independent Bernoulli, like normal eps-greedy).
        explore_mask = self.rng.random(self.n_agents) < epsilon

        # Step 2: only if at least one agent explores, draw correlated uniforms.
        if explore_mask.any():
            Sigma = self._correlation_matrix(obs_list)
            u = self._sample_correlated_uniforms(Sigma)
        else:
            u = None

        actions = []
        for i in range(self.n_agents):
            if explore_mask[i]:
                assert u is not None
                actions.append(int(np.floor(u[i] * self.n_actions)))
            else:
                actions.append(int(np.argmax(q_values_list[i])))
        return actions
