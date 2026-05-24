"""Sweep AE policy constants against the local Novice environment.

This runs the manager directly instead of through HTTP, so it is much faster
than `til test ae` for comparing policy constants. Use the best printed env
vars when building the final Docker image.
"""

from __future__ import annotations

import itertools
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

import ae_manager as policy  # noqa: E402


GRID = {
    "MISSION_MULT": [1.2, 1.5, 1.8],
    "RESOURCE_MULT": [1.2, 1.6, 2.0],
    "RECON_MULT": [0.4, 0.6, 0.8],
    "VISIT_PENALTY": [0.05, 0.10, 0.15],
    "BOMB_BASE_BONUS": [24.0, 35.0, 45.0],
    "MAX_PLAN_DEPTH": [32, 40, 48],
    "DESTRUCTIBLE_OPEN_THRESHOLD": [3.0, 5.0, 8.0],
    "BASE_RUSH_STEP": [45, 60, 90],
    "GOOD_COLLECTIBLE_SCORE": [0.5, 0.75, 1.0],
    "FIXED_MISS_LIMIT": [1, 2, 3],
}


def _jsonable_observation(observation: dict) -> dict:
    return {
        key: value if type(value) in (int, float) else value.tolist()
        for key, value in observation.items()
    }


def _apply_config(config: dict[str, float]) -> None:
    policy.MISSION_MULT = float(config["MISSION_MULT"])
    policy.RESOURCE_MULT = float(config["RESOURCE_MULT"])
    policy.RECON_MULT = float(config["RECON_MULT"])
    policy.VISIT_PENALTY = float(config["VISIT_PENALTY"])
    policy.BOMB_BASE_BONUS = float(config["BOMB_BASE_BONUS"])
    policy.MAX_PLAN_DEPTH = int(config["MAX_PLAN_DEPTH"])
    policy.DESTRUCTIBLE_OPEN_THRESHOLD = float(config["DESTRUCTIBLE_OPEN_THRESHOLD"])
    policy.BASE_RUSH_STEP = int(config["BASE_RUSH_STEP"])
    policy.GOOD_COLLECTIBLE_SCORE = float(config["GOOD_COLLECTIBLE_SCORE"])
    policy.FIXED_MISS_LIMIT = int(config["FIXED_MISS_LIMIT"])
    policy.TARGET_MULTIPLIERS.update(
        {
            "mission": policy.MISSION_MULT,
            "resource": policy.RESOURCE_MULT,
            "recon": policy.RECON_MULT,
        }
    )


def _run_config(config: dict[str, float], rounds: int) -> float:
    from til_environment import bomberman_env
    from til_environment.config import default_config

    _apply_config(config)
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    controlled_agent = env.possible_agents[0]
    manager = policy.AEManager()
    total_reward = 0.0

    for _round in range(rounds):
        env.reset()
        _ = manager.reset()
        round_reward = 0.0
        for agent in env.agent_iter():
            observation, reward, termination, truncation, _info = env.last()
            if agent == controlled_agent:
                round_reward += float(reward)
            if termination or truncation:
                action = None
            elif agent == controlled_agent:
                action = manager.ae(_jsonable_observation(observation))
            else:
                action = env.action_space(agent).sample()
            env.step(action)
        total_reward += round_reward

    env.close()
    return total_reward / max(rounds, 1)


def main() -> None:
    rounds = int(os.getenv("AE_TUNE_ROUNDS", "20"))
    max_trials = int(os.getenv("AE_TUNE_MAX_TRIALS", "120"))
    keys = list(GRID)
    best_score = float("-inf")
    best_config: dict[str, float] | None = None

    for trial_idx, values in enumerate(itertools.product(*(GRID[key] for key in keys)), 1):
        if trial_idx > max_trials:
            break
        config = dict(zip(keys, values))
        score = _run_config(config, rounds)
        print(f"{trial_idx:03d} score={score:.2f} config={config}")
        if score > best_score:
            best_score = score
            best_config = config
            print(f"NEW BEST score={best_score:.2f}")

    print("\nBest average reward:", best_score)
    if best_config:
        for key, value in best_config.items():
            print(f"export AE_{key}={value}")


if __name__ == "__main__":
    main()
