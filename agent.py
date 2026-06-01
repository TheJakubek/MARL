"""VDN learner: per-agent Q-network with sum decomposition Q_tot = sum_i Q_i."""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity: int, n_agents: int):
        self.buf = deque(maxlen=capacity)
        self.n_agents = n_agents

    def push(self, obs, actions, reward, next_obs, done):
        # obs: list of np arrays (one per agent); actions: list of ints
        self.buf.append((obs, actions, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)

        # Stack to (batch, n_agents, obs_dim) etc.
        obs_t = torch.tensor(np.stack([np.stack(o) for o in obs]), dtype=torch.float32)
        next_obs_t = torch.tensor(
            np.stack([np.stack(o) for o in next_obs]), dtype=torch.float32
        )
        actions_t = torch.tensor(np.array(actions), dtype=torch.long)
        rewards_t = torch.tensor(rewards, dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32)
        return obs_t, actions_t, rewards_t, next_obs_t, dones_t

    def __len__(self):
        return len(self.buf)


class VDNLearner:
    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        target_sync: int = 500,
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.target_sync = target_sync
        self.update_count = 0

        self.q_nets = nn.ModuleList([QNet(obs_dim, n_actions) for _ in range(n_agents)])
        self.target_nets = nn.ModuleList(
            [QNet(obs_dim, n_actions) for _ in range(n_agents)]
        )
        self._sync_targets()
        for p in self.target_nets.parameters():
            p.requires_grad_(False)

        self.optimizer = torch.optim.Adam(self.q_nets.parameters(), lr=lr)

    def _sync_targets(self):
        for q, t in zip(self.q_nets, self.target_nets):
            t.load_state_dict(q.state_dict())

    @torch.no_grad()
    def q_values(self, obs_list):
        """obs_list: list of np arrays, one per agent. Returns list of Q-vectors."""
        return [
            self.q_nets[i](torch.tensor(obs_list[i], dtype=torch.float32)).numpy()
            for i in range(self.n_agents)
        ]

    def update(self, batch):
        obs, actions, rewards, next_obs, dones = batch
        # obs: (B, n_agents, obs_dim); actions: (B, n_agents)

        q_tot = 0.0
        q_tot_target = 0.0
        for i in range(self.n_agents):
            q_i = self.q_nets[i](obs[:, i])  # (B, n_actions)
            q_taken = q_i.gather(1, actions[:, i : i + 1]).squeeze(1)  # (B,)
            q_tot = q_tot + q_taken

            with torch.no_grad():
                q_next = self.target_nets[i](next_obs[:, i])  # (B, n_actions)
                q_tot_target = q_tot_target + q_next.max(dim=1).values

        target = rewards + self.gamma * (1 - dones) * q_tot_target
        loss = F.mse_loss(q_tot, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_nets.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_sync == 0:
            self._sync_targets()

        return loss.item()
