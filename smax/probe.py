"""Sanity probe for SMAX 2s3z.

Confirms:
  * jax + jaxmarl import
  * GPU is visible to JAX
  * env can be reset and stepped
  * obs / state / action / reward shapes are what we expect

Run on cluster (or locally) BEFORE writing/running any training code.
"""

import jax
import jax.numpy as jnp
from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario


def main():
    print("=== JAX devices ===")
    print(jax.devices())
    print(f"backend: {jax.default_backend()}")

    print("\n=== Building 2s3z env ===")
    scenario = map_name_to_scenario("2s3z")
    env = make(
        "HeuristicEnemySMAX",
        scenario=scenario,
        use_self_play_reward=False,
        walls_cause_death=True,
        see_enemy_actions=False,
    )
    print(f"agents: {env.agents}")
    print(f"num_agents: {env.num_agents}")

    # Inspect spaces.
    for ag in env.agents:
        os = env.observation_space(ag)
        as_ = env.action_space(ag)
        print(f"  {ag}: obs={os}  act={as_}")

    print("\n=== Reset ===")
    key = jax.random.PRNGKey(0)
    key, k_reset = jax.random.split(key)
    obs, state = env.reset(k_reset)
    for ag, o in obs.items():
        print(f"  obs[{ag}].shape = {o.shape}  dtype={o.dtype}")
    if "world_state" in obs:
        print(f"  world_state.shape = {obs['world_state'].shape}")
    print(f"  state type: {type(state).__name__}")

    print("\n=== Single step (no-op = action 0) ===")
    actions = {ag: jnp.array(0, dtype=jnp.int32) for ag in env.agents}
    key, k_step = jax.random.split(key)
    _, _, rewards, dones, info = env.step(k_step, state, actions)
    print(f"  rewards: {rewards}")
    print(f"  dones: {dones}")
    print(f"  info keys: {list(info.keys()) if hasattr(info, 'keys') else type(info)}")

    print("\n=== Available actions ===")
    avail = env.get_avail_actions(state)
    for ag, a in avail.items():
        print(f"  {ag}: avail={a}  (sum={int(a.sum())})")

    print("\n=== JIT-compiled step ===")
    jit_step = jax.jit(env.step)
    key, k_jit = jax.random.split(key)
    _ = jit_step(k_jit, state, actions)
    print("  jit step compiled and ran ok")

    print("\n=== ALL OK ===")


if __name__ == "__main__":
    main()
