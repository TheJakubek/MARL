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

Keep the $\varepsilon$-greedy template, but **correlate** the random decisions:

1. Build a correlation matrix $R$ (PSD, unit diagonal)
2. Cholesky $R = LL^\top$
3. Sample $z \sim \mathcal{N}(0,I)$, set $y = Lz \sim \mathcal{N}(0,R)$
4. $u_i = \Phi(y_i)$ → uniform marginals, **correlated** across agents

Agent $i$ explores iff $u_i < \varepsilon$. Independent $\varepsilon$-greedy is
exactly the special case $R = I$.

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

All methods reach ~**100% win rate**. Correlated reaches 50% win **~11% (VDN) /
~15% (QMIX) faster** than independent. *(Getting a from-scratch JAX learner to
converge required Double DQN + soft targets + state LayerNorm.)*

---

## Result 4 — what the correlation matrix reveals

![w:880](plot_smax_corr_matrix.png)

**`obs`** keeps role structure (stalker pair 0.80 vs. 0.25 to zealots) — the
observation always encodes unit type. **`q_values`/`hidden`** homogenise
(~0.95+) as the policy converges → why `obs` is the reliable source.

---

## Conclusions

- Correlated $\varepsilon$-greedy via a Gaussian copula is a **simple, drop-in**
  change to exploration
- **Accelerates** learning where coordination is the bottleneck (Hallway +64–94%,
  SMAX 11–15% faster)
- **Improves exploration quality** even where value learning fails (LBF 2.6–2.9×)
- **`obs`** is the most reliable similarity source
- The correlation matrix **recovers role structure** automatically

**Thank you — questions?**
