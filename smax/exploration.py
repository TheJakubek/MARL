"""Exploration in JAX: independent eps-greedy + correlated copula version.

All functions are pure and jittable. They take a PRNGKey, return the actions.

The Gaussian copula construction:
  1. similarity matrix S from per-agent feature vectors (cosine, then clip to [0,1])
  2. add jitter -> PSD correlation matrix R
  3. cholesky factor L
  4. sample z ~ N(0, I), set y = L z       => y ~ N(0, R)
  5. u = Phi(y)                            => uniform marginals with correlation R
  6. each agent's u_i selects between greedy/random based on u_i < eps

Independent eps-greedy is the special case R = I (each agent samples its own
uniform).
"""

from functools import partial

import jax
import jax.numpy as jnp


def _greedy(q: jnp.ndarray, avail: jnp.ndarray) -> jnp.ndarray:
    """argmax over Q masked by avail."""
    masked_q = jnp.where(avail.astype(jnp.bool_), q, -jnp.inf)
    return jnp.argmax(masked_q, axis=-1)


def _random_action(key: jax.Array, avail: jnp.ndarray) -> jnp.ndarray:
    """Sample uniformly from available actions for one agent."""
    # avail is shape (n_actions,) of 0/1.
    logits = jnp.where(avail.astype(jnp.bool_), 0.0, -1e9)
    return jax.random.categorical(key, logits)


def _per_agent_pick(
    key: jax.Array,
    q: jnp.ndarray,         # (n_actions,)
    avail: jnp.ndarray,     # (n_actions,)
    u: jnp.ndarray,         # scalar in [0,1]
    eps: float,
) -> jnp.ndarray:
    """If u < eps -> random masked action. Else greedy."""
    rand_a = _random_action(key, avail)
    greedy_a = _greedy(q, avail)
    return jnp.where(u < eps, rand_a, greedy_a)


def select_independent(
    key: jax.Array,
    q: jnp.ndarray,         # (n_agents, n_actions)
    avail: jnp.ndarray,     # (n_agents, n_actions)
    eps: float,
) -> jnp.ndarray:
    """Standard independent eps-greedy."""
    n = q.shape[0]
    k_u, k_r = jax.random.split(key, 2)
    u = jax.random.uniform(k_u, shape=(n,))
    rkeys = jax.random.split(k_r, n)
    return jax.vmap(_per_agent_pick, in_axes=(0, 0, 0, 0, None))(
        rkeys, q, avail, u, eps
    )


def correlation_matrix(features: jnp.ndarray, jitter: float = 1e-3) -> jnp.ndarray:
    """Cosine similarity over rows -> PSD correlation matrix.

    features: (n_agents, dim).  Returns (n_agents, n_agents).

    We L2-normalize rows then take outer product. Result is in [-1, 1] with
    1's on diagonal. Add jitter*I for numerical PSD.
    """
    norm = jnp.linalg.norm(features, axis=-1, keepdims=True) + 1e-8
    f = features / norm
    R = f @ f.T
    n = R.shape[0]
    return R + jitter * jnp.eye(n)


def sample_correlated_uniforms(
    key: jax.Array,
    R: jnp.ndarray,        # (n, n) correlation matrix, PSD
) -> jnp.ndarray:
    """Sample uniforms with correlation R via Gaussian copula."""
    n = R.shape[0]
    L = jnp.linalg.cholesky(R)               # (n, n) lower triangular
    z = jax.random.normal(key, shape=(n,))   # iid standard normal
    y = L @ z                                 # correlated normals
    # Standard normal CDF.
    u = 0.5 * (1.0 + jax.lax.erf(y / jnp.sqrt(2.0)))
    return u


def select_correlated(
    key: jax.Array,
    q: jnp.ndarray,         # (n_agents, n_actions)
    avail: jnp.ndarray,     # (n_agents, n_actions)
    features: jnp.ndarray,  # (n_agents, feat_dim)  for similarity
    eps: float,
) -> jnp.ndarray:
    """Correlated eps-greedy: agents share a Gaussian-copula uniform draw."""
    R = correlation_matrix(features)
    k_u, k_r = jax.random.split(key, 2)
    u = sample_correlated_uniforms(k_u, R)              # (n,)
    n = q.shape[0]
    rkeys = jax.random.split(k_r, n)
    return jax.vmap(_per_agent_pick, in_axes=(0, 0, 0, 0, None))(
        rkeys, q, avail, u, eps
    )


@partial(jax.jit, static_argnames=("kind",))
def select(
    key: jax.Array,
    q: jnp.ndarray,
    avail: jnp.ndarray,
    features: jnp.ndarray,
    eps: float,
    kind: str = "independent",
) -> jnp.ndarray:
    """Dispatch to independent or correlated."""
    if kind == "independent":
        return select_independent(key, q, avail, eps)
    elif kind == "correlated":
        return select_correlated(key, q, avail, features, eps)
    else:
        raise ValueError(f"unknown exploration kind: {kind}")
