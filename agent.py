"""MARL value-decomposition learners: VDN (sum) and QMIX (monotonic mixer)."""

import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class QNet(nn.Module):
    """Per-agent Q-network. Backbone (obs -> hidden) is separated from the head
    (hidden -> Q) so that the hidden representation can be used as a feature
    vector for similarity-based exploration."""

    def __init__(self, obs_dim: int, n_actions: int, hidden: int = 128):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden, n_actions)
        self.hidden_dim = hidden

    def forward(self, x, return_hidden: bool = False):
        h = self.backbone(x)
        q = self.head(h)
        if return_hidden:
            return q, h
        return q


class SumMixer(nn.Module):
    """VDN: Q_tot = sum_i Q_i. No state input, no parameters."""

    def forward(self, q_per_agent: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # q_per_agent: (B, n_agents) ; state: unused
        del state
        return q_per_agent.sum(dim=1)


class QMixer(nn.Module):
    """QMIX monotonic mixer with state-conditioned hypernetwork.

    Architecture (per Rashid et al. 2018):
      hyper_w1, hyper_b1 produce first-layer mixer weights from state.
      hyper_w2, hyper_b2 produce second-layer mixer weights from state.
      Weights are forced non-negative via abs() to guarantee monotonicity:
        d Q_tot / d Q_i >= 0  for all i.
    """

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 16, hyper_hidden: int = 32):
        super().__init__()
        self.n_agents = n_agents
        self.embed_dim = embed_dim

        # Hypernet: state -> first-layer weights (n_agents -> embed_dim)
        self.hyper_w1 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, n_agents * embed_dim),
        )
        self.hyper_b1 = nn.Linear(state_dim, embed_dim)

        # Hypernet: state -> second-layer weights (embed_dim -> 1)
        self.hyper_w2 = nn.Sequential(
            nn.Linear(state_dim, hyper_hidden),
            nn.ReLU(),
            nn.Linear(hyper_hidden, embed_dim),
        )
        # Bias of last layer: deeper hypernet (paper-faithful), produces a scalar V(s).
        self.hyper_b2 = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

    def forward(self, q_per_agent: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        # q_per_agent: (B, n_agents) ; state: (B, state_dim)
        B = q_per_agent.shape[0]

        # First mixer layer: (B, 1, n_agents) @ (B, n_agents, embed_dim) -> (B, 1, embed_dim)
        w1 = torch.abs(self.hyper_w1(state)).view(B, self.n_agents, self.embed_dim)
        b1 = self.hyper_b1(state).view(B, 1, self.embed_dim)
        q_in = q_per_agent.view(B, 1, self.n_agents)
        hidden = F.elu(torch.bmm(q_in, w1) + b1)  # (B, 1, embed_dim)

        # Second mixer layer: (B, 1, embed_dim) @ (B, embed_dim, 1) -> (B, 1, 1)
        w2 = torch.abs(self.hyper_w2(state)).view(B, self.embed_dim, 1)
        b2 = self.hyper_b2(state).view(B, 1, 1)
        q_tot = (torch.bmm(hidden, w2) + b2).view(B)  # (B,)
        return q_tot


class ReplayBuffer:
    def __init__(self, capacity: int, n_agents: int):
        self.buf = deque(maxlen=capacity)
        self.n_agents = n_agents

    def push(self, obs, actions, reward, next_obs, done):
        self.buf.append((obs, actions, reward, next_obs, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buf, batch_size)
        obs, actions, rewards, next_obs, dones = zip(*batch)
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

    def success_count(self) -> int:
        """How many transitions in the buffer have a positive reward."""
        return sum(1 for (_, _, r, _, _) in self.buf if r > 0)


class BalancedReplayBuffer:
    """Two-pool replay: a main pool (everything) and a success pool (transitions
    with reward > 0). Sampling mixes them with a fixed ratio to prevent the
    rare-success problem.
    """

    def __init__(self, capacity: int, n_agents: int, success_fraction: float = 0.25):
        self.main = ReplayBuffer(capacity=capacity, n_agents=n_agents)
        self.success = ReplayBuffer(capacity=capacity // 4, n_agents=n_agents)
        self.success_fraction = success_fraction
        self.n_agents = n_agents

    def push(self, obs, actions, reward, next_obs, done):
        self.main.push(obs, actions, reward, next_obs, done)
        if reward > 0:
            self.success.push(obs, actions, reward, next_obs, done)

    def sample(self, batch_size: int):
        n_success = int(batch_size * self.success_fraction)
        # Only oversample if we actually have success transitions.
        if len(self.success) == 0:
            n_success = 0
        n_main = batch_size - n_success

        if n_success == 0:
            return self.main.sample(n_main)

        s_obs, s_act, s_r, s_no, s_d = self.success.sample(min(n_success, len(self.success)))
        m_obs, m_act, m_r, m_no, m_d = self.main.sample(n_main + max(0, n_success - len(self.success)))

        return (
            torch.cat([s_obs, m_obs], dim=0),
            torch.cat([s_act, m_act], dim=0),
            torch.cat([s_r, m_r], dim=0),
            torch.cat([s_no, m_no], dim=0),
            torch.cat([s_d, m_d], dim=0),
        )

    def __len__(self):
        return len(self.main)

    def success_count(self) -> int:
        return len(self.success)


class MARLLearner:
    """Value-decomposition learner. Mixer is pluggable: SumMixer (VDN) or QMixer.

    parameter_sharing: if True, all agents use the SAME QNet, with a one-hot
    agent-id appended to each observation (so the network can still differentiate
    agents). This is the standard setup in modern MARL codebases (EPyMARL, PyMARL).
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        n_actions: int,
        mixer_kind: str = "vdn",
        lr: float = 1e-4,
        gamma: float = 0.99,
        target_sync: int = 500,
        device: str | None = None,
        parameter_sharing: bool = False,
    ):
        self.n_agents = n_agents
        self.n_actions = n_actions
        self.gamma = gamma
        self.target_sync = target_sync
        self.update_count = 0
        self.mixer_kind = mixer_kind
        self.parameter_sharing = parameter_sharing

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)

        # Effective input dim = obs + agent_id one-hot (only when sharing).
        eff_obs_dim = obs_dim + n_agents if parameter_sharing else obs_dim
        self.eff_obs_dim = eff_obs_dim

        if parameter_sharing:
            shared_q = QNet(eff_obs_dim, n_actions)
            shared_target = QNet(eff_obs_dim, n_actions)
            # Replicate the SAME module n_agents times so the rest of the code
            # (which iterates over q_nets[i]) stays untouched.
            self.q_nets = nn.ModuleList([shared_q] * n_agents)
            self.target_nets = nn.ModuleList([shared_target] * n_agents)
        else:
            self.q_nets = nn.ModuleList([QNet(eff_obs_dim, n_actions) for _ in range(n_agents)])
            self.target_nets = nn.ModuleList(
                [QNet(eff_obs_dim, n_actions) for _ in range(n_agents)]
            )

        # State for mixer = concatenation of all agents' obs (proxy for global state).
        state_dim = obs_dim * n_agents
        self.mixer: nn.Module
        self.target_mixer: nn.Module
        if mixer_kind == "vdn":
            self.mixer = SumMixer()
            self.target_mixer = SumMixer()
        elif mixer_kind == "qmix":
            self.mixer = QMixer(n_agents=n_agents, state_dim=state_dim)
            self.target_mixer = QMixer(n_agents=n_agents, state_dim=state_dim)
        else:
            raise ValueError(f"unknown mixer_kind: {mixer_kind}")

        self.q_nets.to(self.device)
        self.target_nets.to(self.device)
        self.mixer.to(self.device)
        self.target_mixer.to(self.device)

        self._sync_targets()
        for p in self.target_nets.parameters():
            p.requires_grad_(False)
        for p in self.target_mixer.parameters():
            p.requires_grad_(False)

        # Deduplicate parameters when parameter_sharing is True (q_nets[0] is
        # q_nets[1] is ... -> nn.ModuleList iterates each entry, would double-count).
        seen_ids = set()
        q_params = []
        for q in self.q_nets:
            if id(q) in seen_ids:
                continue
            seen_ids.add(id(q))
            q_params.extend(q.parameters())
        params = q_params + list(self.mixer.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def _sync_targets(self):
        # With parameter sharing, q_nets[0] == q_nets[1] (same object), so we
        # only need to copy once. The loop below handles both cases correctly:
        # for sharing, it copies the same state_dict twice (no-op second time).
        seen = set()
        for q, t in zip(self.q_nets, self.target_nets):
            if id(q) in seen:
                continue
            t.load_state_dict(q.state_dict())
            seen.add(id(q))
        if self.mixer_kind == "qmix":
            self.target_mixer.load_state_dict(self.mixer.state_dict())

    def _augment_obs(self, obs_tensor: torch.Tensor, agent_idx: int) -> torch.Tensor:
        """If parameter sharing, append a one-hot agent-id to obs.
        obs_tensor shape: (..., obs_dim) -> (..., obs_dim + n_agents)
        """
        if not self.parameter_sharing:
            return obs_tensor
        one_hot = torch.zeros(self.n_agents, device=obs_tensor.device, dtype=obs_tensor.dtype)
        one_hot[agent_idx] = 1.0
        # Broadcast one_hot to match obs_tensor's batch dims.
        target_shape = list(obs_tensor.shape[:-1]) + [self.n_agents]
        one_hot = one_hot.expand(target_shape)
        return torch.cat([obs_tensor, one_hot], dim=-1)

    @torch.no_grad()
    def q_values(self, obs_list, return_hidden: bool = False):
        """Return per-agent Q-vectors. If return_hidden, also return per-agent
        hidden activations (the post-backbone representation)."""
        qs = []
        hs = []
        for i in range(self.n_agents):
            x = torch.tensor(obs_list[i], dtype=torch.float32, device=self.device)
            x = self._augment_obs(x, i)
            q, h = self.q_nets[i](x, return_hidden=True)
            qs.append(q.cpu().numpy())
            hs.append(h.cpu().numpy())
        if return_hidden:
            return qs, hs
        return qs

    def update(self, batch):
        obs, actions, rewards, next_obs, dones = batch
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_obs = next_obs.to(self.device)
        dones = dones.to(self.device)
        # obs: (B, n_agents, obs_dim); actions: (B, n_agents); rewards/dones: (B,)
        B = obs.shape[0]

        q_taken_list = []
        q_next_max_list = []
        for i in range(self.n_agents):
            obs_i = self._augment_obs(obs[:, i], i)
            next_obs_i = self._augment_obs(next_obs[:, i], i)
            q_i = self.q_nets[i](obs_i)
            q_taken = q_i.gather(1, actions[:, i : i + 1]).squeeze(1)
            q_taken_list.append(q_taken)
            with torch.no_grad():
                q_next = self.target_nets[i](next_obs_i)
                q_next_max_list.append(q_next.max(dim=1).values)

        q_taken_all = torch.stack(q_taken_list, dim=1)  # (B, n_agents)
        q_next_max_all = torch.stack(q_next_max_list, dim=1)  # (B, n_agents)

        # State proxy = concat of obs.
        state = obs.reshape(B, -1)
        next_state = next_obs.reshape(B, -1)

        q_tot = self.mixer(q_taken_all, state)
        with torch.no_grad():
            q_tot_target = self.target_mixer(q_next_max_all, next_state)
            target = rewards + self.gamma * (1 - dones) * q_tot_target

        loss = F.mse_loss(q_tot, target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            [p for g in self.optimizer.param_groups for p in g["params"]],
            max_norm=10.0,
        )
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_sync == 0:
            self._sync_targets()

        return loss.item()


# Backward-compat alias for any older imports.
VDNLearner = MARLLearner
