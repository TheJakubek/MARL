"""Thin adapter around gymnasium lb-foraging env to match our interface:
    reset() -> [obs_per_agent]
    step(actions: list[int]) -> (obs, shared_reward, done, info)
"""

import gymnasium as gym
import lbforaging  # noqa: F401  (registers envs)
import numpy as np


class LBFAdapter:
    def __init__(self, env_id: str = "Foraging-6x6-2p-1f-coop-v3", seed: int | None = None):
        self.env = gym.make(env_id)
        self.seed = seed
        self.n_agents = len(self.env.action_space)  # type: ignore[arg-type]
        space0 = self.env.action_space[0]  # type: ignore[index]
        self.n_actions = int(space0.n)  # type: ignore[attr-defined]
        obs_space0 = self.env.observation_space[0]  # type: ignore[index]
        self.obs_dim = int(obs_space0.shape[0])  # type: ignore[attr-defined]

    def reset(self):
        obs, _ = self.env.reset(seed=self.seed)
        self.seed = None  # only seed the very first reset
        return [np.asarray(o, dtype=np.float32) for o in obs]

    def step(self, actions):
        obs, rewards, terminated, truncated, info = self.env.step(tuple(int(a) for a in actions))
        shared_r = float(sum(rewards)) if isinstance(rewards, (list, tuple, np.ndarray)) else float(rewards)
        done = bool(terminated or truncated)
        info = dict(info) if isinstance(info, dict) else {}
        info["success"] = shared_r > 0.0
        obs = [np.asarray(o, dtype=np.float32) for o in obs]
        return obs, shared_r, done, info

    @property
    def N_AGENTS(self):
        return self.n_agents

    @property
    def N_ACTIONS(self):
        return self.n_actions
