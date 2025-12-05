"""
A simple test script to check if the RecfgEnv environment resets correctly.
"""

import numpy as np
import os
import sys

# Ensure the project root is in the path to import leo_gym
# This is a bit of a hack for running a script from a subdirectory
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.append(project_root)

from src.leo_gym.gyms.recfg_gym import RecfgEnv
from notebooks.C4_Reconfiguration.train_recfg_hppo_cfg import env_cfg

def test_recfg_env_reset():
    """
    Tests if the RecfgEnv resets correctly and if the initial
    deviations are properly randomized and seeded.
    """
    print("--- Testing RecfgEnv reset and randomization ---")

    # 1. Create the environment with an initial seed
    # The environment's RNG is seeded once at initialization.
    env = RecfgEnv(cfg=env_cfg, seed=42)
    print("✅ Environment created successfully.")

    # 2. First reset (without a seed)
    # This will use the environment's internal, advancing RNG state.
    _, _ = env.reset()
    roe1 = env.satellite.roe[0]
    print(f"✅ First reset successful. Initial ROE: {np.round(roe1, 2)}")

    # 3. Second reset (should be different)
    # This will continue using the internal RNG, producing a new state.
    _, _ = env.reset()
    roe2 = env.satellite.roe[0]
    print(f"✅ Second reset successful. Initial ROE: {np.round(roe2, 2)}")

    # The randomized components (indices 4 and 5) should differ.
    if np.allclose(roe1[4:], roe2[4:]):
        print("❌ The ROE deviations are the same after two unseeded resets. Randomization might not be working as expected.")
    else:
        print("✅ The ROE deviations are different after two unseeded resets, as expected.")

    # 4. Test deterministic reset with a specific seed
    print("\n--- Testing deterministic reset with seeding ---")
    SEED = 123
    _, _ = env.reset(seed=SEED)
    roe_seeded1 = env.satellite.roe[0]
    print(f"✅ Reset with seed={SEED} successful. Initial ROE: {np.round(roe_seeded1, 2)}")

    # Reset again with the exact same seed
    _, _ = env.reset(seed=SEED)
    roe_seeded2 = env.satellite.roe[0]
    print(f"✅ Reset again with seed={SEED} successful. Initial ROE: {np.round(roe_seeded2, 2)}")

    if np.allclose(roe_seeded1, roe_seeded2):
        print("✅ The ROE deviations are identical for resets with the same seed, as expected.")
    else:
        print("❌ The ROE deviations are different for resets with the same seed. Seeding is not working correctly.")

    # 5. Test that a different seed gives a different result
    print("\n--- Testing with a different seed ---")
    _, _ = env.reset(seed=SEED + 1)
    roe_new_seed = env.satellite.roe[0]
    print(f"✅ Reset with seed={SEED + 1} successful. Initial ROE: {np.round(roe_new_seed, 2)}")

    if np.allclose(roe_seeded1, roe_new_seed):
        print("❌ The ROE deviations are the same for different seeds.")
    else:
        print("✅ The ROE deviations are different for different seeds, as expected.")

    print("\n🎉 Test completed successfully! 🎉")

if __name__ == "__main__":
    test_recfg_env_reset()
