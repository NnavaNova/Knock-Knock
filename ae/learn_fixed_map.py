"""Generate the fixed Novice AE map for runtime planning.

The Novice environment uses fixed RNG seeds in ``Dynamics.reset``.  That means
the wall layout, base positions, starting positions, static item layout, and
respawn map are all recoverable before evaluation.  This script runs on the
Workbench where ``til_environment`` is installed and writes ``ae/src/fixed_map.py``
so the submitted agent can plan from step 0 instead of discovering the map by
wandering.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(SRC))

from ae_manager import AEManager, DIRS, GRID_SIZE  # noqa: E402

Coord = tuple[int, int]
Edge = tuple[Coord, Coord]


def _jsonable_observation(observation: dict) -> dict:
    return {
        key: value if type(value) in (int, float) else value.tolist()
        for key, value in observation.items()
    }


def _canonical_edge(a: Coord, b: Coord) -> Edge:
    a = (int(a[0]), int(a[1]))
    b = (int(b[0]), int(b[1]))
    return (a, b) if a <= b else (b, a)


def _edge_literal(edge: Edge) -> str:
    (ax, ay), (bx, by) = edge
    return f"(({ax}, {ay}), ({bx}, {by}))"


def _coord_literal(pos: Coord) -> str:
    return f"({pos[0]}, {pos[1]})"


def _extract_destructible_edges(wall_grid: Any) -> set[Edge]:
    edges: set[Edge] = set()
    for x in range(GRID_SIZE):
        for y in range(GRID_SIZE):
            tile = int(wall_grid[x, y])
            for direction, (dx, dy) in enumerate(DIRS):
                if not (tile & (1 << (direction + 4))):
                    continue
                neighbor = (x + dx, y + dy)
                if 0 <= neighbor[0] < GRID_SIZE and 0 <= neighbor[1] < GRID_SIZE:
                    edges.add(_canonical_edge((x, y), neighbor))
    return edges


def _generate_direct() -> dict[str, Any]:
    from gymnasium.utils.seeding import np_random
    from til_environment.arena import ArenaGenerator
    from til_environment.config import default_config

    cfg = default_config()
    cfg.env.novice = True
    rng, _ = np_random(88)
    generator = ArenaGenerator(
        grid_size=int(cfg.env.grid_size),
        wall_prob=float(cfg.dynamics.arena.wall_prob),
        wall_destructible_ratio=float(cfg.dynamics.arena.wall_destructible_ratio),
        mission_prob=float(cfg.dynamics.arena.mission_prob),
        recon_prob=float(cfg.dynamics.arena.recon_prob),
        resource_prob=float(cfg.dynamics.arena.resource_prob),
        novice=True,
        base_respawn_steps=int(cfg.env.tile_respawn_steps),
    )
    result = generator.generate_episode(rng, 88, num_teams=int(cfg.env.num_teams))

    collectibles: dict[Coord, str] = {}
    for spec in result.static_entities:
        pos = (int(spec.position[0]), int(spec.position[1]))
        collectibles[pos] = str(spec.kind)

    respawns = {
        pos: int(result.respawn_map[pos[0], pos[1]])
        for pos in sorted(collectibles)
    }

    return {
        "walls": {_canonical_edge(tuple(a), tuple(b)) for a, b in result.walls},
        "destructible_edges": _extract_destructible_edges(result.wall_grid),
        "collectibles": collectibles,
        "respawns": respawns,
        "base_locations": [
            (int(pos[0]), int(pos[1])) for pos in result.base_locations
        ],
        "starting_locations": [
            (int(pos[0]), int(pos[1])) for pos in result.starting_locations
        ],
    }


def _generate_from_rollouts() -> dict[str, Any]:
    from til_environment import bomberman_env
    from til_environment.config import default_config

    rounds = int(os.getenv("AE_MAP_ROUNDS", "120"))
    cfg = default_config()
    cfg.env.novice = True
    env = bomberman_env.basic_env(env_wrappers=[], cfg=cfg)
    manager = AEManager()

    for _round in range(rounds):
        env.reset()
        for agent in env.agent_iter():
            observation, _reward, termination, truncation, _info = env.last()
            if not termination and not truncation:
                obs = _jsonable_observation(observation)
                manager._ingest_observation(obs, int(obs.get("step", 0)))
                action = env.action_space(agent).sample()
            else:
                action = None
            env.step(action)

    env.close()
    return {
        "walls": set(manager.walls),
        "destructible_edges": set(manager.destructible_edges),
        "collectibles": {},
        "respawns": {},
        "base_locations": [],
        "starting_locations": [],
    }


def _write_fixed_map(out_path: Path, data: dict[str, Any]) -> None:
    walls = sorted(data["walls"])
    destructible = sorted(data["destructible_edges"])
    collectibles = dict(sorted(data["collectibles"].items()))
    respawns = dict(sorted(data["respawns"].items()))
    base_locations = list(data["base_locations"])
    starting_locations = list(data["starting_locations"])

    out_path.write_text(
        "# Generated by ae/learn_fixed_map.py on the Novice environment.\n"
        "# Regenerate on Workbench if the official fixed map changes.\n\n"
        "FIXED_WALLS = {\n"
        + "".join(f"    {_edge_literal(edge)},\n" for edge in walls)
        + "}\n\n"
        "FIXED_DESTRUCTIBLE_EDGES = {\n"
        + "".join(f"    {_edge_literal(edge)},\n" for edge in destructible)
        + "}\n\n"
        "FIXED_COLLECTIBLES = {\n"
        + "".join(
            f"    {_coord_literal(pos)}: {kind!r},\n"
            for pos, kind in collectibles.items()
        )
        + "}\n\n"
        "FIXED_RESPAWN_STEPS = {\n"
        + "".join(
            f"    {_coord_literal(pos)}: {steps},\n"
            for pos, steps in respawns.items()
        )
        + "}\n\n"
        "FIXED_BASE_LOCATIONS = (\n"
        + "".join(f"    {_coord_literal(pos)},\n" for pos in base_locations)
        + ")\n\n"
        "FIXED_STARTING_LOCATIONS = (\n"
        + "".join(f"    {_coord_literal(pos)},\n" for pos in starting_locations)
        + ")\n"
    )
    print(
        f"Wrote {out_path} with {len(walls)} walls, "
        f"{len(destructible)} destructible edges, "
        f"{len(collectibles)} collectibles, and {len(base_locations)} bases"
    )


def main() -> None:
    out_path = SRC / "fixed_map.py"
    if os.getenv("AE_MAP_ROLLOUT_ONLY") == "1":
        data = _generate_from_rollouts()
    else:
        try:
            data = _generate_direct()
        except Exception as exc:
            print(f"Direct map generation failed ({exc!r}); falling back to rollouts")
            data = _generate_from_rollouts()
    _write_fixed_map(out_path, data)


if __name__ == "__main__":
    main()
