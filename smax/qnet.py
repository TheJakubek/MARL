"""Shared QNet for SMAX 2s3z (5 agents, 2 unit types).

Single network, parameter-shared across all agents. Disambiguates agents via
two one-hot vectors concatenated to the obs:

  * agent_id one-hot (5-dim)         distinguishes within-role copies
  * role one-hot (2-dim: stalker/zealot)

This lets the backbone exploit shared structure ("attack visible enemy") while
the head can still produce role-aware Q-values.

Returns Q-values per action. With `return_hidden=True` also returns the
backbone hidden state (used by the "hidden" similarity source for the copula).
"""

from typing import Sequence

import flax.linen as nn
import jax.numpy as jnp


class QNet(nn.Module):
    """MLP backbone + linear head."""
    n_actions: int
    hidden_sizes: Sequence[int] = (128, 128)

    @nn.compact
    def __call__(self, x: jnp.ndarray, return_hidden: bool = False):
        h = x
        for size in self.hidden_sizes:
            h = nn.Dense(size)(h)
            h = nn.relu(h)
        q = nn.Dense(self.n_actions)(h)
        if return_hidden:
            return q, h
        return q


def augment_obs(
    obs: jnp.ndarray,           # (n_agents, obs_dim)
    agent_id_oh: jnp.ndarray,   # (n_agents, n_agents)
    role_oh: jnp.ndarray,       # (n_agents, n_roles)
) -> jnp.ndarray:
    """Concatenate (obs, agent_id_one_hot, role_one_hot)."""
    return jnp.concatenate([obs, agent_id_oh, role_oh], axis=-1)
