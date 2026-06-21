---
marp: true
theme: default
paginate: true
size: 16:9
---

<!-- _paginate: false -->

# Correlated Exploration for Cooperative MARL

### via a Gaussian Copula

**Project 9 — Reinforcement Learning**

---

## The problem: coordinated exploration

In cooperative MARL the team is often rewarded **only** when agents act
*together* (e.g. both reach a goal in the same step).

- Standard $\varepsilon$-greedy explores each agent **independently**
- Probability all $n$ agents explore the matching joint action $\propto \varepsilon^{n}$
- → decays **exponentially** in the number of agents

**Hypothesis:** coupling agents' exploration — so similar agents tend to act
alike — should accelerate learning on coordination-bottlenecked tasks.

---

## Method: Gaussian copula

Keep the $\varepsilon$-greedy template, but **correlate which action** the
exploring agents take (who explores stays an independent Bernoulli):

1. Build a correlation matrix $R$ (PSD, unit diagonal)
2. Cholesky $R = LL^\top$
3. Sample $z \sim \mathcal{N}(0,I)$, set $y = Lz \sim \mathcal{N}(0,R)$
4. $u_i = \Phi(y_i)$ → uniform marginals, **correlated** across agents

An exploring agent takes action $\lfloor u_i\cdot|A_i|\rfloor$. Correlated
agents draw similar $u_i$ → **same joint action** (focus-fire, both on switches).
Independent $\varepsilon$-greedy is the special case $R = I$.

---

## Where does the correlation come from?

$R$ = **cosine similarity** of per-agent feature vectors → a Gram matrix,
automatically PSD with unit diagonal (+ small jitter).

Three similarity sources compared:

- **`obs`** — the agent's observation
- **`q_values`** — its current Q-values
- **`hidden`** — its Q-network backbone activations

**Role-agnostic:** nothing tells the method which agents share a role — similar
agents simply get correlated automatically.

---

## Experimental setup

**Environments**
- **Hallway** (MAVEN) — 2 agents, separate corridors, reward only if both reach 0
- **Level-Based Foraging** — 2 agents must load food *together* (sparse)
- **SMAX `2s3z`** (StarCraft-like, JAX) — 5 agents, 2 stalkers + 3 zealots

**Algorithms:** VDN (sum) and QMIX (monotonic mixer), parameter-shared
Q-network, DQN-style training, **5 seeds** each.

---

## Result 1 — Hallway: faster early learning

![w:620](plot_hallway_vdn.png)

Correlated-`obs` early success (ep 50–100): **+64%** (VDN), **+94%** (QMIX) over
independent. All converge eventually → the benefit is a **speed-up**.

---

## Result 2 — LBF: the mechanism works even where learning fails

![w:560](plot_lbf_vdn.png)

Catastrophic forgetting collapses all methods, **but** correlated reaches a
**2.6–2.9× higher peak** — it finds the coordinated successes independent
exploration misses.

---

## Result 3 — SMAX 2s3z: scales to GPU

![w:560](plot_smax_qmix.png)

Under **QMIX**, correlated lifts the **final win rate 0.81 → 0.91** (and reaches
50% sooner); under the easier VDN all reach ~0.95, correlated ~6% faster.
*(Convergence required Double DQN + soft targets + a LayerNorm on the SMAX state.)*

---

## Result 4 — *when* does correlation encode roles?

![w:620](plot_smax_corr_evolution.png)

Role contrast = within-role − cross-role correlation. **`obs`** stays positive
through the exploration-heavy phase (always encodes unit type); **`q_values`**
is noisy, **`hidden`** weak → why `obs` is the reliable source.

---

## Conclusions

- Correlating agents' exploratory **actions** via a Gaussian copula is a
  **simple, drop-in** change to exploration
- **Accelerates** learning where coordination is the bottleneck (Hallway +64–94%)
- **Improves exploration quality** even where value learning fails (LBF 2.6–2.9×)
- **Scales to GPU (SMAX):** lifts QMIX final win rate **0.81 → 0.91**
- **`obs`** is the most reliable similarity source; couples situationally-similar
  agents (stalkers most strongly) with no role info supplied

**Thank you — questions?**
