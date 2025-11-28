## C4 Reconfiguration (R/T/N) – How to use

Everything here targets the R/T/N reconfiguration gym (`RecfgEnv`: hybrid discrete+continuous actions).

- `train_recfg_hppo_cfg.py` – build the environment and HPPO configs. Set action bounds, thrust (`f_max`), horizons, and seeds here. Outputs `env_cfg` and `ppo_cfg` used by training.
- `train_recfg.ipynb` – run training. Loads the configs above, spawns parallel `RecfgEnv` workers, and runs HPPO; logs to MLflow if enabled. Open the notebook, set seeds/paths, run all cells.
- `openlooptest_recfg.ipynb` – quick sanity test without a policy. Define a scripted list of burns (axis_id, delay, duration), run it, and view ROE evolution. Set `SEED` to fix or randomize the initial orbit.

Core flow
1) Configure: tweak `train_recfg_hppo_cfg.py` (ROE dynamics, action bounds, thrust, reward targets/seeds).
2) Train: run `train_recfg.ipynb`; it builds vectorized `RecfgEnv`, collects rollouts (`steps_per_env` per worker), and updates the HPPO policy.
3) Validate open-loop: use `openlooptest_recfg.ipynb` to see how scripted R/T/N plans change ROE.

Notes
- Action space: `discrete` = {0:+R,1:-R,2:+T,3:-T,4:+N,5:-N,6:coast}; `continuous` = [delay_steps, duration_steps].
- Reward: axis-agnostic shaping toward a target ROE plus fuel penalty (see `RecfgEnv`).
- Seeding: set `SEED` to an int for deterministic runs; `None` for random starts. 