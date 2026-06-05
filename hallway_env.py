"""Hallway environment from MAVEN (Mahajan et al. 2019, arXiv:1910.07483).

Two agents on two separate corridors of (possibly different) lengths. Each
agent only observes its own position (partial observability — does NOT see
the other agent). Goal: both agents simultaneously at position 0 in the
same timestep.

This is the canonical toy benchmark for coordinated exploration in MARL:
without correlated exploration the probability of joint success decays
exponentially with corridor length.
"""

import numpy as np


class Hallway:
    N_AGENTS = 2
    N_ACTIONS = 3  # 0=left, 1=stay, 2=right

    def __init__(self, lengths=(3, 4), max_steps: int | None = None, seed=None):
        """lengths: (L1, L2) — length of each corridor (number of cells = L+1,
        positions 0..L). Default (3, 4) matches MAVEN's smallest setup."""
        self.lengths = tuple(lengths)
        assert len(self.lengths) == self.N_AGENTS
        self.max_L = max(self.lengths)
        self.max_steps = max_steps if max_steps is not None else 4 * self.max_L
        self.rng = np.random.default_rng(seed)
        self.positions: list[int] = []
        self.t = 0

    def reset(self):
        # MAVEN starts each agent at the middle of their corridor.
        self.positions = [L // 2 for L in self.lengths]
        self.t = 0
        return self._obs()

    def step(self, actions):
        assert len(actions) == self.N_AGENTS
        deltas = {0: -1, 1: 0, 2: 1}

        for i, a in enumerate(actions):
            new_pos = self.positions[i] + deltas[int(a)]
            new_pos = max(0, min(self.lengths[i], new_pos))
            self.positions[i] = new_pos

        self.t += 1
        success = all(p == 0 for p in self.positions)
        reward = 1.0 if success else 0.0
        done = success or self.t >= self.max_steps
        return self._obs(), reward, done, {"success": success}

    def _obs(self):
        # Each agent sees only its own (normalized) position. Partial obs.
        # Shape per agent: (1,).
        return [
            np.array([self.positions[i] / self.max_L], dtype=np.float32)
            for i in range(self.N_AGENTS)
        ]

    @property
    def obs_dim(self):
        return 1
