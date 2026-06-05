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

    def select(self, q_values_list, obs_list, epsilon, hidden_list=None):
        del obs_list, hidden_list  # unused; kept for interface parity
        actions = []
        for i in range(self.n_agents):
            if self.rng.random() < epsilon:
                actions.append(int(self.rng.integers(self.n_actions)))
            else:
                actions.append(int(np.argmax(q_values_list[i])))
        return actions


class CorrelatedEpsilonGreedy:
    """Random exploration actions sampled from a Gaussian copula. The copula's
    correlation matrix is the cosine-similarity Gram matrix of per-agent feature
    vectors. Which features to use is configurable.

    similarity_source:
      "obs"      -> raw observations (correlates agents in similar world positions)
      "q_values" -> Q-vectors (correlates agents whose policies prefer similar
                    actions; targets *intent* rather than position)
      "hidden"   -> Q-network's penultimate-layer activations (correlates agents
                    in semantically similar situations as encoded by the learned
                    representation)
    """

    SOURCES = {"obs", "q_values", "hidden"}

    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        rng=None,
        jitter: float = 1e-4,
        similarity_source: str = "obs",
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.rng = rng or np.random.default_rng()
        self.jitter = jitter
        if similarity_source not in self.SOURCES:
            raise ValueError(f"unknown similarity_source: {similarity_source}")
        self.similarity_source = similarity_source

    def _features(self, q_values_list, obs_list, hidden_list):
        if self.similarity_source == "obs":
            return np.stack(obs_list, axis=0)
        if self.similarity_source == "q_values":
            return np.stack(q_values_list, axis=0)
        if hidden_list is None:
            raise ValueError(
                "similarity_source='hidden' requires hidden_list to be passed to select()"
            )
        return np.stack(hidden_list, axis=0)

    def _correlation_matrix(self, features: np.ndarray) -> np.ndarray:
        # L2-normalize rows, take outer product -> Gram matrix (PSD by construction).
        norms = np.linalg.norm(features, axis=1, keepdims=True) + 1e-8
        Xn = features / norms
        Sigma = Xn @ Xn.T
        Sigma = Sigma + self.jitter * np.eye(self.n_agents)
        return Sigma

    def _sample_correlated_uniforms(self, Sigma):
        L = np.linalg.cholesky(Sigma)
        z_iid = self.rng.standard_normal(self.n_agents)
        z = L @ z_iid  # correlated standard normals
        u = norm.cdf(z)  # marginal Uniform(0,1), still correlated
        return u

    def select(self, q_values_list, obs_list, epsilon, hidden_list=None):
        # Step 1: decide who explores (independent Bernoulli, same exploration
        # budget as the baseline; we only correlate *which* random action they pick).
        explore_mask = self.rng.random(self.n_agents) < epsilon

        # Step 2: only if at least one agent explores, draw correlated uniforms.
        if explore_mask.any():
            features = self._features(q_values_list, obs_list, hidden_list)
            Sigma = self._correlation_matrix(features)
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
