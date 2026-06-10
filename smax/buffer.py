"""Flat replay buffer for SMAX training.

We store one transition at a time (not whole episodes — VDN/QMIX can learn
from individual transitions). Stored fields:

  obs:        (cap, n_agents, obs_dim)
  state:      (cap, state_dim)            # global, for QMIX
  actions:    (cap, n_agents)
  rewards:    (cap,)                       # team reward (shared)
  next_obs:   (cap, n_agents, obs_dim)
  next_state: (cap, state_dim)
  next_avail: (cap, n_agents, n_actions)
  done:       (cap,)

Implementation: numpy-backed (host memory). For our scale (cap ~100k) this is
fine and avoids JAX donation/aliasing complexity.
"""

import numpy as np


class ReplayBuffer:
    def __init__(self, cap: int, n_agents: int, obs_dim: int,
                 state_dim: int, n_actions: int):
        self.cap = cap
        self.idx = 0
        self.size = 0

        self.obs = np.zeros((cap, n_agents, obs_dim), dtype=np.float32)
        self.state = np.zeros((cap, state_dim), dtype=np.float32)
        self.actions = np.zeros((cap, n_agents), dtype=np.int32)
        self.rewards = np.zeros((cap,), dtype=np.float32)
        self.next_obs = np.zeros((cap, n_agents, obs_dim), dtype=np.float32)
        self.next_state = np.zeros((cap, state_dim), dtype=np.float32)
        self.next_avail = np.zeros((cap, n_agents, n_actions), dtype=np.int32)
        self.done = np.zeros((cap,), dtype=np.float32)

    def add(self, obs, state, actions, reward, next_obs, next_state, next_avail, done):
        i = self.idx
        self.obs[i] = obs
        self.state[i] = state
        self.actions[i] = actions
        self.rewards[i] = reward
        self.next_obs[i] = next_obs
        self.next_state[i] = next_state
        self.next_avail[i] = next_avail
        self.done[i] = float(done)
        self.idx = (self.idx + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, batch_size: int, rng: np.random.Generator):
        assert self.size > 0, "buffer is empty"
        idxs = rng.integers(0, self.size, size=batch_size)
        return dict(
            obs=self.obs[idxs],
            state=self.state[idxs],
            actions=self.actions[idxs],
            rewards=self.rewards[idxs],
            next_obs=self.next_obs[idxs],
            next_state=self.next_state[idxs],
            next_avail=self.next_avail[idxs],
            done=self.done[idxs],
        )
