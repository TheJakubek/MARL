"""Coordinated Switches: 2-agent gridworld where both agents must stand on
their assigned switches in the same timestep to get a reward.

Designed as a minimal toy environment for MARL exploration experiments.
"""

import numpy as np


class CoordinatedSwitches:
    GRID_SIZE = 5
    N_AGENTS = 2
    N_ACTIONS = 5  # 0=stay, 1=N, 2=S, 3=E, 4=W
    MAX_STEPS = 30
    STEP_PENALTY = -0.01
    SUCCESS_REWARD = 1.0

    # Switch positions (one per agent), fixed across episodes.
    SWITCHES = [(0, 4), (4, 0)]

    def __init__(self, seed=None):
        self.rng = np.random.default_rng(seed)
        self.agent_pos: list[tuple[int, int]] = []
        self.t = 0

    def reset(self):
        # Random non-overlapping start positions for both agents.
        positions = self.rng.choice(
            self.GRID_SIZE * self.GRID_SIZE, size=self.N_AGENTS, replace=False
        )
        self.agent_pos = [
            (int(p // self.GRID_SIZE), int(p % self.GRID_SIZE)) for p in positions
        ]
        self.t = 0
        return self._obs()

    def step(self, actions):
        assert len(actions) == self.N_AGENTS
        deltas = {0: (0, 0), 1: (-1, 0), 2: (1, 0), 3: (0, 1), 4: (0, -1)}

        for i, a in enumerate(actions):
            dr, dc = deltas[int(a)]
            r, c = self.agent_pos[i]
            nr = max(0, min(self.GRID_SIZE - 1, r + dr))
            nc = max(0, min(self.GRID_SIZE - 1, c + dc))
            self.agent_pos[i] = (nr, nc)

        self.t += 1

        on_switch = [self.agent_pos[i] == self.SWITCHES[i] for i in range(self.N_AGENTS)]
        success = all(on_switch)

        reward = self.SUCCESS_REWARD if success else self.STEP_PENALTY
        done = success or self.t >= self.MAX_STEPS

        return self._obs(), reward, done, {"success": success}

    def _obs(self):
        # Each agent sees: own (row, col), teammate (row, col), own switch (row, col).
        # Normalized to [0, 1]. Shape per agent: (6,).
        g = self.GRID_SIZE - 1
        obs = []
        for i in range(self.N_AGENTS):
            j = 1 - i
            own = self.agent_pos[i]
            mate = self.agent_pos[j]
            sw = self.SWITCHES[i]
            obs.append(
                np.array(
                    [own[0] / g, own[1] / g, mate[0] / g, mate[1] / g, sw[0] / g, sw[1] / g],
                    dtype=np.float32,
                )
            )
        return obs

    @property
    def obs_dim(self):
        return 6
