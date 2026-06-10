"""Value mixers in Flax: VDN sum + QMIX hypernetwork."""

import flax.linen as nn
import jax.numpy as jnp


class VDNMixer(nn.Module):
    """Q_tot = sum_i Q_i. Stateless."""

    @nn.compact
    def __call__(self, q_chosen: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        # q_chosen: (..., n_agents). state unused (kept for unified API).
        del state
        return q_chosen.sum(axis=-1, keepdims=True)


class QMixMixer(nn.Module):
    """Monotonic QMIX with a hypernetwork conditioned on global state."""
    n_agents: int
    embed_dim: int = 32
    hyper_hidden: int = 64

    @nn.compact
    def __call__(self, q_chosen: jnp.ndarray, state: jnp.ndarray) -> jnp.ndarray:
        # q_chosen: (B, n_agents)   (or (n_agents,) if single sample)
        # state:    (B, state_dim)  (or (state_dim,))
        # Return:   (B, 1)          (or (1,))
        single = q_chosen.ndim == 1
        if single:
            q_chosen = q_chosen[None]   # (1, n_agents)
            state = state[None]         # (1, state_dim)

        # Hypernet: state -> w1 (n_agents x embed_dim), b1 (embed_dim)
        w1 = nn.Dense(self.hyper_hidden)(state)
        w1 = nn.relu(w1)
        w1 = nn.Dense(self.n_agents * self.embed_dim)(w1)
        w1 = jnp.abs(w1)                                 # monotonic
        w1 = w1.reshape(-1, self.n_agents, self.embed_dim)

        b1 = nn.Dense(self.embed_dim)(state)             # (B, embed_dim)

        # Hypernet: state -> w2 (embed_dim x 1), b2 (1)
        w2 = nn.Dense(self.hyper_hidden)(state)
        w2 = nn.relu(w2)
        w2 = nn.Dense(self.embed_dim)(w2)
        w2 = jnp.abs(w2).reshape(-1, self.embed_dim, 1)  # monotonic

        b2 = nn.Dense(self.hyper_hidden)(state)
        b2 = nn.relu(b2)
        b2 = nn.Dense(1)(b2)                             # (B, 1)

        # Forward through mixer.
        q_in = q_chosen[:, None, :]                       # (B, 1, n_agents)
        h = jnp.matmul(q_in, w1) + b1[:, None, :]         # (B, 1, embed_dim)
        h = nn.elu(h)
        out = jnp.matmul(h, w2) + b2[:, None, :]          # (B, 1, 1)
        out = out.squeeze(axis=1)                         # (B, 1)

        if single:
            out = out.squeeze(axis=0)                      # (1,)
        return out
