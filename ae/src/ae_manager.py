"""Stateful planner for the TIL AE Bomberman task.

The local baseline only reacted to objects currently visible in the agent
viewcone.  This version keeps an absolute-coordinate world model, plans over
position and facing direction, and treats bombs as timed hazards.  It is still
small enough for fast inference: every decision runs a few BFS searches over a
16x16x4 state space.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
import os
from typing import Any, Callable, Iterable

import numpy as np


# Actions from til_environment.actions.Action.
FORWARD = 0
BACKWARD = 1
LEFT = 2
RIGHT = 3
STAY = 4
PLACE_BOMB = 5
NUM_ACTIONS = 6

# Directions from til_environment.types.Direction.
DIRS: tuple[tuple[int, int], ...] = (
    (1, 0),   # RIGHT
    (0, 1),   # DOWN
    (-1, 0),  # LEFT
    (0, -1),  # UP
)

# ViewChannel indices from til_environment.observation.ViewChannel.
CH_VISIBLE = 0
CH_WALL_R, CH_WALL_D, CH_WALL_L, CH_WALL_U = 1, 2, 3, 4
CH_TILE_EMPTY, CH_TILE_RECON, CH_TILE_MISSION, CH_TILE_RESOURCE = 5, 6, 7, 8
CH_ALLY_AGENT, CH_ENEMY_AGENT, CH_ALLY_BASE, CH_ENEMY_BASE = 9, 10, 11, 12
CH_DWALL_R, CH_DWALL_D, CH_DWALL_L, CH_DWALL_U = 13, 14, 15, 16
CH_ALLY_BOMB, CH_ENEMY_BOMB = 17, 18
CH_ALLY_BOMB_TIMER, CH_ENEMY_BOMB_TIMER = 19, 20
CH_ALLY_AGENT_HEALTH, CH_ENEMY_AGENT_HEALTH = 21, 22
CH_ALLY_BASE_HEALTH, CH_ENEMY_BASE_HEALTH = 23, 24

GRID_SIZE = 16
AGENT_ROW = 2
AGENT_COL = 2
BOMB_TIMER = 4
BLAST_RADIUS = 2
MAX_PLAN_DEPTH = 48
STALE_ENEMY_STEPS = 4
STALE_COLLECTIBLE_STEPS = 70
RESPAWN_STEPS = 40

MISSION_MULT = float(os.getenv("AE_MISSION_MULT", "1.5"))
RESOURCE_MULT = float(os.getenv("AE_RESOURCE_MULT", "1.6"))
RECON_MULT = float(os.getenv("AE_RECON_MULT", "0.6"))
VISIT_PENALTY = float(os.getenv("AE_VISIT_PENALTY", "0.05"))
BOMB_BASE_BONUS = float(os.getenv("AE_BOMB_BASE_BONUS", "35"))
DESTRUCTIBLE_OPEN_THRESHOLD = float(os.getenv("AE_DESTRUCTIBLE_OPEN_THRESHOLD", "5"))
BASE_RUSH_STEP = int(os.getenv("AE_BASE_RUSH_STEP", "60"))
CLOSE_COMBAT_DISTANCE = int(os.getenv("AE_CLOSE_COMBAT_DISTANCE", "6"))
GOOD_COLLECTIBLE_SCORE = float(os.getenv("AE_GOOD_COLLECTIBLE_SCORE", "0.75"))
LOW_BOMB_RESOURCE_BONUS = float(os.getenv("AE_LOW_BOMB_RESOURCE_BONUS", "1.0"))
FIXED_MISS_LIMIT = int(os.getenv("AE_FIXED_MISS_LIMIT", "2"))

Coord = tuple[int, int]
Edge = tuple[Coord, Coord]

WALL_CHANNELS = (CH_WALL_R, CH_WALL_D, CH_WALL_L, CH_WALL_U)
DWALL_CHANNELS = (CH_DWALL_R, CH_DWALL_D, CH_DWALL_L, CH_DWALL_U)

COLLECTIBLE_VALUES = {
    "mission": 5.0,
    "resource": 2.0,
    "recon": 1.0,
}

TARGET_MULTIPLIERS = {
    "mission": MISSION_MULT,
    "resource": RESOURCE_MULT,
    "recon": RECON_MULT,
}

COLLECTIBLE_CHANNELS = (
    (CH_TILE_MISSION, "mission"),
    (CH_TILE_RESOURCE, "resource"),
    (CH_TILE_RECON, "recon"),
)


@dataclass
class BombInfo:
    timer: int
    owner: str
    last_seen: int


@dataclass
class TargetInfo:
    kind: str
    value: float
    last_seen: int


@dataclass
class SearchResult:
    first_action: int
    distance: int
    end_pos: Coord
    end_dir: int


class AEManager:
    """Stateful search policy for one evaluation container."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.walls: set[Edge] = set()
        self.destructible_edges: set[Edge] = set()
        self.seen_cells: set[Coord] = set()
        self.collectibles: dict[Coord, TargetInfo] = {}
        self.known_collectibles: dict[Coord, TargetInfo] = {}
        self.fixed_collectibles: set[Coord] = set()
        self.fixed_respawns: dict[Coord, int] = {}
        self.fixed_base_locations: set[Coord] = set()
        self.fixed_collectible_misses: defaultdict[Coord, int] = defaultdict(int)
        self.dismissed_fixed_bases: set[Coord] = set()
        self.collected_until: dict[Coord, int] = {}
        self.ally_bases: set[Coord] = set()
        self.enemy_bases: dict[Coord, TargetInfo] = {}
        self.enemies: dict[Coord, TargetInfo] = {}
        self.bombs: dict[Coord, BombInfo] = {}
        self.visited: defaultdict[Coord, int] = defaultdict(int)
        self.temporary_blocks: dict[Edge, int] = {}
        self.previous_location: Coord | None = None
        self.previous_direction: int = 0
        self.previous_action: int | None = None
        self.last_step: int | None = None
        self.team_bombs = 0
        self.team_resources = 0.0
        self.just_placed_bomb = False
        self.stuck_steps = 0
        self._load_fixed_map()

    def ae(self, observation: dict[str, Any]) -> int:
        step = self._read_int(observation.get("step"), 0)
        if self.last_step is not None and step < self.last_step:
            self.reset()
        if step == 0:
            self.reset()

        location = self._read_location(observation)
        direction = self._read_int(observation.get("direction"), 0) % 4
        action_mask = self._read_action_mask(observation)
        team_bombs = self._read_int(observation.get("team_bombs"), 0)
        team_resources = self._read_float(observation.get("team_resources"), 0.0)
        self.team_bombs = team_bombs
        self.team_resources = team_resources

        self._age_state(step)
        self._record_failed_move(location, direction, step)
        self._ingest_observation(observation, step)
        self._mark_collected(location, step)
        self._trim_stale_state(step)

        if location is None:
            action = self._first_legal((STAY, RIGHT, LEFT, FORWARD), action_mask)
        elif self._is_frozen(observation, action_mask):
            action = self._first_legal((STAY,), action_mask)
        else:
            self.visited[location] += 1
            self.stuck_steps = (
                self.stuck_steps + 1 if location == self.previous_location else 0
            )
            action = self._choose_action(
                location=location,
                direction=direction,
                action_mask=action_mask,
                team_bombs=team_bombs,
                step=step,
            )

        self.previous_location = location
        self.previous_direction = direction
        self.previous_action = action
        self.last_step = step
        self.just_placed_bomb = action == PLACE_BOMB
        return int(action)

    # ------------------------------------------------------------------
    # Observation parsing and map updates

    def _ingest_observation(self, observation: dict[str, Any], step: int) -> None:
        location = self._read_location(observation)
        direction = self._read_int(observation.get("direction"), 0) % 4
        agent_view = self._read_array(observation.get("agent_viewcone"))
        if agent_view is None:
            agent_view = self._read_array(observation.get("viewcone"))
        if location is not None and agent_view is not None and agent_view.ndim == 3:
            self._ingest_agent_view(agent_view, location, direction, step)

        base_location = self._read_location(observation, "base_location")
        if base_location is not None:
            self.ally_bases.add(base_location)
            self._activate_fixed_bases(base_location, step)
        base_view = self._read_array(observation.get("base_viewcone"))
        if base_location is not None and base_view is not None and base_view.ndim == 3:
            self._ingest_base_view(base_view, base_location, step)

    def _ingest_agent_view(
        self, view: np.ndarray, location: Coord, direction: int, step: int
    ) -> None:
        rows, cols, channels = view.shape
        if channels < 25:
            return
        origin_row = min(AGENT_ROW, rows - 1)
        origin_col = min(AGENT_COL, cols - 1)
        for row in range(rows):
            for col in range(cols):
                cell = view[row, col]
                if not self._cell_is_visible(cell):
                    continue
                rel = (row - origin_row, col - origin_col)
                world = self._agent_view_to_world(location, direction, rel)
                if self._in_bounds(world):
                    self._ingest_cell(world, cell, step)

    def _ingest_base_view(self, view: np.ndarray, base_location: Coord, step: int) -> None:
        rows, cols, channels = view.shape
        if channels < 25:
            return
        origin_row = rows // 2
        origin_col = cols // 2
        for row in range(rows):
            for col in range(cols):
                cell = view[row, col]
                if not self._cell_is_visible(cell):
                    continue
                world = (
                    base_location[0] + row - origin_row,
                    base_location[1] + col - origin_col,
                )
                if self._in_bounds(world):
                    self._ingest_cell(world, cell, step)

    def _ingest_cell(self, pos: Coord, cell: np.ndarray, step: int) -> None:
        self.seen_cells.add(pos)
        self._update_edges(pos, cell)
        self._update_collectibles(pos, cell, step)
        self._update_entities(pos, cell, step)
        self._update_bombs(pos, cell, step)

    def _update_edges(self, pos: Coord, cell: np.ndarray) -> None:
        for direction in range(4):
            edge = self._edge_in_direction(pos, direction)
            if edge is None:
                continue
            has_wall = cell[WALL_CHANNELS[direction]] > 0.5
            has_dwall = cell[DWALL_CHANNELS[direction]] > 0.5
            if has_wall or has_dwall:
                self.walls.add(edge)
            else:
                self.walls.discard(edge)
            if has_dwall:
                self.destructible_edges.add(edge)
            else:
                self.destructible_edges.discard(edge)

    def _update_collectibles(self, pos: Coord, cell: np.ndarray, step: int) -> None:
        found: TargetInfo | None = None
        for channel, kind in COLLECTIBLE_CHANNELS:
            if cell[channel] > 0.5:
                found = TargetInfo(
                    kind=kind,
                    value=COLLECTIBLE_VALUES[kind],
                    last_seen=step,
                )
                break
        if found is None:
            previous = self.collectibles.pop(pos, None)
            known = previous or self.known_collectibles.get(pos)
            if known is not None:
                if pos in self.fixed_collectibles:
                    self.fixed_collectible_misses[pos] += 1
                    if self.fixed_collectible_misses[pos] >= FIXED_MISS_LIMIT:
                        self.fixed_collectibles.discard(pos)
                        self.known_collectibles.pop(pos, None)
                        self.collected_until.pop(pos, None)
                        return
                self.known_collectibles[pos] = known
                self.collected_until[pos] = max(
                    self.collected_until.get(pos, 0), step + self._respawn_steps(pos)
                )
        else:
            self.collectibles[pos] = found
            self.known_collectibles[pos] = found
            self.fixed_collectible_misses.pop(pos, None)
            self.collected_until.pop(pos, None)

    def _update_entities(self, pos: Coord, cell: np.ndarray, step: int) -> None:
        if cell[CH_ALLY_BASE] > 0.5:
            self.ally_bases.add(pos)

        if cell[CH_ENEMY_BASE] > 0.5:
            health = float(cell[CH_ENEMY_BASE_HEALTH] or 1.0)
            self.enemy_bases[pos] = TargetInfo("enemy_base", 80.0 + 40.0 * health, step)
            self.dismissed_fixed_bases.discard(pos)
        else:
            self.enemy_bases.pop(pos, None)
            if pos in self.fixed_base_locations:
                self.dismissed_fixed_bases.add(pos)

        if cell[CH_ENEMY_AGENT] > 0.5:
            health = float(cell[CH_ENEMY_AGENT_HEALTH] or 1.0)
            self.enemies[pos] = TargetInfo("enemy", 30.0 + 15.0 * (1.0 - health), step)
        else:
            self.enemies.pop(pos, None)

    def _update_bombs(self, pos: Coord, cell: np.ndarray, step: int) -> None:
        has_ally = cell[CH_ALLY_BOMB] > 0.5
        has_enemy = cell[CH_ENEMY_BOMB] > 0.5
        if has_ally or has_enemy:
            timer = max(float(cell[CH_ALLY_BOMB_TIMER]), float(cell[CH_ENEMY_BOMB_TIMER]))
            if timer <= 0:
                timer = 1
            self.bombs[pos] = BombInfo(
                timer=int(round(timer)),
                owner="ally" if has_ally else "enemy",
                last_seen=step,
            )
        else:
            self.bombs.pop(pos, None)

    def _age_state(self, step: int) -> None:
        if self.last_step is None:
            return
        elapsed = max(0, step - self.last_step)
        if elapsed <= 0:
            return
        for pos in list(self.bombs):
            bomb = self.bombs[pos]
            bomb.timer -= elapsed
            if bomb.timer < 0:
                self.bombs.pop(pos, None)
        for edge, until in list(self.temporary_blocks.items()):
            if until <= step:
                self.temporary_blocks.pop(edge, None)

    def _trim_stale_state(self, step: int) -> None:
        for pos, target in list(self.enemies.items()):
            if step - target.last_seen > STALE_ENEMY_STEPS:
                self.enemies.pop(pos, None)
        for pos, target in list(self.collectibles.items()):
            if step - target.last_seen > STALE_COLLECTIBLE_STEPS:
                self.collectibles.pop(pos, None)
        for pos, until in list(self.collected_until.items()):
            if until + STALE_COLLECTIBLE_STEPS < step:
                self.collected_until.pop(pos, None)

    def _mark_collected(self, location: Coord | None, step: int) -> None:
        if location is None:
            return
        known = self.collectibles.pop(location, None)
        if known is not None:
            self.known_collectibles[location] = known
            self.collected_until[location] = step + self._respawn_steps(location)

    def _record_failed_move(
        self, location: Coord | None, direction: int, step: int
    ) -> None:
        if (
            self.previous_location is None
            or location is None
            or self.previous_action not in (FORWARD, BACKWARD)
        ):
            return
        expected_pos, _ = self._transition(
            self.previous_location, self.previous_direction, self.previous_action
        )
        if expected_pos != self.previous_location and location == self.previous_location:
            edge = self._edge_between(self.previous_location, expected_pos)
            if edge is not None:
                self.temporary_blocks[edge] = step + 6

    # ------------------------------------------------------------------
    # Decision policy

    def _choose_action(
        self,
        location: Coord,
        direction: int,
        action_mask: list[int],
        team_bombs: int,
        step: int,
    ) -> int:
        if self._danger_now(location):
            escape = self._plan_to_safety(location, direction, action_mask)
            if escape is not None:
                return escape

        if self._should_place_bomb(location, direction, action_mask, team_bombs, step):
            return PLACE_BOMB

        target_action = self._best_target_action(location, direction, action_mask, step)
        if target_action is not None:
            return target_action

        frontier = self._plan_to_frontier(location, direction, action_mask, step)
        if frontier is not None:
            return frontier

        return self._safe_fallback(location, direction, action_mask)

    def _should_place_bomb(
        self,
        location: Coord,
        direction: int,
        action_mask: list[int],
        team_bombs: int,
        step: int,
    ) -> bool:
        if (
            team_bombs <= 0
            or not self._legal(action_mask, PLACE_BOMB)
            or self.just_placed_bomb
            or self._danger_now(location)
        ):
            return False
        if not self._escape_exists_after_bomb(location, direction):
            return False

        for pos in self.enemy_bases:
            if self._bomb_hits(location, pos):
                return True
        for pos in self.enemies:
            if self._bomb_hits(location, pos):
                return True

        if self.stuck_steps >= 4:
            edge = self._best_destructible_edge_to_open(location, direction, step)
            if edge is not None and any(self._bomb_hits(location, p) for p in edge):
                return True
        return False

    def _best_target_action(
        self, location: Coord, direction: int, action_mask: list[int], step: int
    ) -> int | None:
        candidates: list[tuple[float, Coord, str]] = []
        reachable = self._bfs_reachable(
            location,
            direction,
            action_mask,
            max_depth=MAX_PLAN_DEPTH,
            avoid_danger=True,
        )

        for base_pos, info in self.enemy_bases.items():
            for pos in self._bombing_positions(base_pos):
                if self._in_bounds(pos) and not self._cell_blocked(pos):
                    candidates.append(
                        (
                            self._combat_target_value("enemy_base", step),
                            pos,
                            "enemy_base",
                        )
                    )

        for base_pos in self._suspected_enemy_bases():
            for pos in self._bombing_positions(base_pos):
                if self._in_bounds(pos) and not self._cell_blocked(pos):
                    candidates.append(
                        (
                            self._combat_target_value("suspected_enemy_base", step),
                            pos,
                            "suspected_enemy_base",
                        )
                    )

        for enemy_pos, info in self.enemies.items():
            for pos in self._bombing_positions(enemy_pos):
                if self._in_bounds(pos) and not self._cell_blocked(pos):
                    candidates.append(
                        (self._combat_target_value("enemy", step), pos, "enemy")
                    )

        for pos, info in self._available_collectibles(step).items():
            age_penalty = (
                0.0
                if pos in self.fixed_collectibles
                else max(0, step - info.last_seen) * 0.03
            )
            multiplier = TARGET_MULTIPLIERS.get(info.kind, 1.0)
            if info.kind == "resource" and self.team_bombs <= 1:
                multiplier += LOW_BOMB_RESOURCE_BONUS
            candidates.append((info.value * multiplier - age_penalty, pos, info.kind))

        best_score = -1e9
        best: SearchResult | None = None
        best_kind = ""
        best_collectible_score = -1e9
        best_collectible: SearchResult | None = None
        for value, target, kind in candidates:
            result = self._best_reachable_result(reachable, target)
            if result is None:
                continue
            adjusted_value = value
            if kind in ("enemy_base", "enemy") and self._escape_exists_after_bomb(
                result.end_pos, result.end_dir
            ):
                adjusted_value += BOMB_BASE_BONUS
            novelty_bonus = 1.0 if self._has_unknown_neighbor(target) else 0.0
            risk_penalty = 1.0 if self._danger_at(target, result.distance + 1) else 0.0
            score = (
                adjusted_value / (result.distance + 1.0)
                + 0.20 * novelty_bonus
                - VISIT_PENALTY * self.visited[target]
                - 4.0 * risk_penalty
            )
            if (
                kind not in ("enemy_base", "enemy", "suspected_enemy_base")
                and score > best_collectible_score
            ):
                best_collectible_score = score
                best_collectible = result
            if score > best_score:
                best_score = score
                best = result
                best_kind = kind

        if best is not None and best_score > 0.05:
            if (
                best_kind in ("enemy_base", "enemy", "suspected_enemy_base")
                and step < BASE_RUSH_STEP
                and best.distance > CLOSE_COMBAT_DISTANCE
                and best_collectible is not None
                and best_collectible_score >= GOOD_COLLECTIBLE_SCORE
            ):
                return best_collectible.first_action
            return best.first_action
        return None

    def _plan_to_frontier(
        self, location: Coord, direction: int, action_mask: list[int], step: int
    ) -> int | None:
        def frontier_goal(pos: Coord, _direction: int) -> bool:
            if pos not in self.seen_cells:
                return True
            if self.visited[pos] == 0 and self._has_unknown_neighbor(pos):
                return True
            return self._has_unknown_neighbor(pos) and self.visited[pos] <= 1

        result = self._search(
            location,
            direction,
            action_mask,
            goal=frontier_goal,
            max_depth=MAX_PLAN_DEPTH,
            avoid_danger=True,
            action_order=(FORWARD, LEFT, RIGHT, BACKWARD),
        )
        if result is not None:
            return result.first_action

        # If every reachable frontier is known blocked, drift toward the least
        # visited legal area to avoid loops.
        least_visited = self._search(
            location,
            direction,
            action_mask,
            goal=lambda p, _d: self.visited[p] == 0,
            max_depth=MAX_PLAN_DEPTH,
            avoid_danger=True,
            action_order=(FORWARD, RIGHT, LEFT, BACKWARD),
        )
        return least_visited.first_action if least_visited is not None else None

    def _plan_to_safety(
        self, location: Coord, direction: int, action_mask: list[int]
    ) -> int | None:
        result = self._search(
            location,
            direction,
            action_mask,
            goal=lambda p, _d: not self._danger_at(p, 1),
            max_depth=10,
            avoid_danger=True,
            action_order=(FORWARD, BACKWARD, LEFT, RIGHT, STAY),
        )
        return result.first_action if result is not None else None

    def _safe_fallback(
        self, location: Coord, direction: int, action_mask: list[int]
    ) -> int:
        fallback_scores: list[tuple[float, int]] = []
        for action in (FORWARD, RIGHT, LEFT, BACKWARD, STAY):
            if not self._legal(action_mask, action):
                continue
            next_pos, _ = self._transition(location, direction, action)
            if next_pos != location and self._movement_blocked(location, next_pos):
                continue
            if not self._danger_at(next_pos, 1):
                return action
            fallback_scores.append((self._bomb_distance_score(next_pos, action), action))
        if fallback_scores:
            return max(fallback_scores)[1]
        return self._first_legal((STAY, RIGHT, LEFT, BACKWARD, FORWARD), action_mask)

    # ------------------------------------------------------------------
    # Search and geometry

    def _search(
        self,
        start_pos: Coord,
        start_dir: int,
        action_mask: list[int],
        goal: Callable[[Coord, int], bool],
        max_depth: int,
        avoid_danger: bool,
        action_order: tuple[int, ...] = (FORWARD, BACKWARD, LEFT, RIGHT),
        extra_bombs: Iterable[tuple[Coord, int]] = (),
    ) -> SearchResult | None:
        queue: deque[tuple[Coord, int, int, int | None]] = deque()
        queue.append((start_pos, start_dir, 0, None))
        seen: set[tuple[Coord, int]] = {(start_pos, start_dir)}
        extra = tuple(extra_bombs)

        while queue:
            pos, direction, depth, first_action = queue.popleft()
            if depth > 0 and goal(pos, direction):
                return SearchResult(
                    first_action=first_action if first_action is not None else STAY,
                    distance=depth,
                    end_pos=pos,
                    end_dir=direction,
                )
            if depth >= max_depth:
                continue

            for action in action_order:
                if depth == 0 and not self._legal(action_mask, action):
                    continue
                next_pos, next_dir = self._transition(pos, direction, action)
                if next_pos != pos and self._movement_blocked(pos, next_pos):
                    continue
                next_depth = depth + 1
                if avoid_danger and self._danger_at(next_pos, next_depth, extra):
                    continue
                state = (next_pos, next_dir)
                if state in seen:
                    continue
                seen.add(state)
                queue.append(
                    (
                        next_pos,
                        next_dir,
                        next_depth,
                        action if first_action is None else first_action,
                    )
                )
        return None

    def _bfs_reachable(
        self,
        start_pos: Coord,
        start_dir: int,
        action_mask: list[int],
        max_depth: int,
        avoid_danger: bool = True,
    ) -> dict[tuple[Coord, int], SearchResult]:
        queue: deque[tuple[Coord, int, int, int | None]] = deque()
        queue.append((start_pos, start_dir, 0, None))
        seen: set[tuple[Coord, int]] = {(start_pos, start_dir)}
        best: dict[tuple[Coord, int], SearchResult] = {}

        while queue:
            pos, direction, depth, first_action = queue.popleft()
            if depth > 0:
                best.setdefault(
                    (pos, direction),
                    SearchResult(
                        first_action=first_action if first_action is not None else STAY,
                        distance=depth,
                        end_pos=pos,
                        end_dir=direction,
                    ),
                )
            if depth >= max_depth:
                continue

            for action in (FORWARD, BACKWARD, LEFT, RIGHT, STAY):
                if depth == 0 and not self._legal(action_mask, action):
                    continue
                next_pos, next_dir = self._transition(pos, direction, action)
                if next_pos != pos and self._movement_blocked(pos, next_pos):
                    continue
                next_depth = depth + 1
                if avoid_danger and self._danger_at(next_pos, next_depth):
                    continue
                state = (next_pos, next_dir)
                if state in seen:
                    continue
                seen.add(state)
                queue.append(
                    (
                        next_pos,
                        next_dir,
                        next_depth,
                        action if first_action is None else first_action,
                    )
                )
        return best

    @staticmethod
    def _best_reachable_result(
        reachable: dict[tuple[Coord, int], SearchResult], target: Coord
    ) -> SearchResult | None:
        matches = [result for (pos, _direction), result in reachable.items() if pos == target]
        if not matches:
            return None
        return min(matches, key=lambda result: result.distance)

    def _transition(self, pos: Coord, direction: int, action: int) -> tuple[Coord, int]:
        direction %= 4
        if action == LEFT:
            return pos, (direction - 1) % 4
        if action == RIGHT:
            return pos, (direction + 1) % 4
        if action == FORWARD:
            dx, dy = DIRS[direction]
            return (pos[0] + dx, pos[1] + dy), direction
        if action == BACKWARD:
            dx, dy = DIRS[(direction + 2) % 4]
            return (pos[0] + dx, pos[1] + dy), direction
        return pos, direction

    def _movement_blocked(self, src: Coord, dst: Coord) -> bool:
        if not self._in_bounds(dst):
            return True
        edge = self._edge_between(src, dst)
        if edge is None:
            return True
        if edge in self.walls or edge in self.temporary_blocks:
            return True
        if self._cell_blocked(dst):
            return True
        return False

    def _cell_blocked(self, pos: Coord) -> bool:
        if pos in self.enemy_bases or pos in self.ally_bases or pos in self.bombs:
            return True
        enemy = self.enemies.get(pos)
        return enemy is not None and self.last_step is not None and (
            self.last_step - enemy.last_seen <= 1
        )

    def _edge_in_direction(self, pos: Coord, direction: int) -> Edge | None:
        dx, dy = DIRS[direction]
        return self._edge_between(pos, (pos[0] + dx, pos[1] + dy))

    def _edge_between(self, a: Coord, b: Coord) -> Edge | None:
        if not self._in_bounds(a) or not self._in_bounds(b):
            return None
        return (a, b) if a <= b else (b, a)

    @staticmethod
    def _agent_view_to_world(location: Coord, direction: int, rel: Coord) -> Coord:
        dr, dc = rel
        if direction == 0:  # RIGHT
            return location[0] + dr, location[1] + dc
        if direction == 1:  # DOWN
            return location[0] - dc, location[1] + dr
        if direction == 2:  # LEFT
            return location[0] - dr, location[1] - dc
        return location[0] + dc, location[1] - dr  # UP

    def _bombing_positions(self, target: Coord) -> list[Coord]:
        positions: list[Coord] = []
        tx, ty = target
        for x in range(tx - BLAST_RADIUS, tx + BLAST_RADIUS + 1):
            for y in range(ty - BLAST_RADIUS, ty + BLAST_RADIUS + 1):
                pos = (x, y)
                if self._in_bounds(pos) and self._bomb_hits(pos, target):
                    positions.append(pos)
        return positions

    def _available_collectibles(self, step: int) -> dict[Coord, TargetInfo]:
        available = dict(self.collectibles)
        for pos, info in self.known_collectibles.items():
            if pos in available:
                continue
            if self.collected_until.get(pos, -1) <= step:
                available[pos] = info
        return available

    def _best_destructible_edge_to_open(
        self, location: Coord, direction: int, step: int
    ) -> Edge | None:
        best_edge: Edge | None = None
        best_gain = 0.0
        for edge in self.destructible_edges:
            if not any(self._bomb_hits(location, pos) for pos in edge):
                continue
            gain = self._edge_opens_value(edge, location, direction, step)
            if gain > best_gain:
                best_gain = gain
                best_edge = edge
        if best_gain > DESTRUCTIBLE_OPEN_THRESHOLD:
            return best_edge
        return None

    def _edge_opens_value(
        self, edge: Edge, location: Coord, direction: int, step: int
    ) -> float:
        before = self._reachable_target_value(location, direction, step, depth=16)
        had_wall = edge in self.walls
        had_dwall = edge in self.destructible_edges
        self.walls.discard(edge)
        self.destructible_edges.discard(edge)
        try:
            after = self._reachable_target_value(location, direction, step, depth=16)
        finally:
            if had_wall:
                self.walls.add(edge)
            if had_dwall:
                self.destructible_edges.add(edge)
        return after - before

    def _reachable_target_value(
        self, location: Coord, direction: int, step: int, depth: int
    ) -> float:
        reachable = self._bfs_reachable(
            location,
            direction,
            [1, 1, 1, 1, 1, 0],
            max_depth=depth,
            avoid_danger=True,
        )
        value = 0.0
        for pos, info in self._available_collectibles(step).items():
            result = self._best_reachable_result(reachable, pos)
            if result is not None:
                value += (
                    info.value
                    * TARGET_MULTIPLIERS.get(info.kind, 1.0)
                    / (result.distance + 1.0)
                )
        for base_pos in self.enemy_bases:
            for pos in self._bombing_positions(base_pos):
                result = self._best_reachable_result(reachable, pos)
                if result is not None:
                    value += (
                        self._combat_target_value("enemy_base", step)
                        / (result.distance + 1.0)
                    )
                    break
        for base_pos in self._suspected_enemy_bases():
            for pos in self._bombing_positions(base_pos):
                result = self._best_reachable_result(reachable, pos)
                if result is not None:
                    value += (
                        self._combat_target_value("suspected_enemy_base", step)
                        / (result.distance + 1.0)
                    )
                    break
        return value

    def _escape_exists_after_bomb(self, location: Coord, direction: int) -> bool:
        future_bomb = ((location, BOMB_TIMER),)
        result = self._search(
            location,
            direction,
            [1, 1, 1, 1, 1, 0],
            goal=lambda p, _d: not self._bomb_hits(location, p)
            and not self._danger_at(p, 1, future_bomb),
            max_depth=BOMB_TIMER,
            avoid_danger=True,
            action_order=(FORWARD, BACKWARD, LEFT, RIGHT),
            extra_bombs=future_bomb,
        )
        return result is not None

    def _danger_now(self, pos: Coord) -> bool:
        for bomb_pos, bomb in self.bombs.items():
            if self._bomb_hits(bomb_pos, pos) and bomb.timer <= 2:
                return True
        return False

    def _danger_at(
        self,
        pos: Coord,
        dt: int,
        extra_bombs: Iterable[tuple[Coord, int]] = (),
    ) -> bool:
        for bomb_pos, bomb in self.bombs.items():
            if self._bomb_hits(bomb_pos, pos) and bomb.timer - dt == 0:
                return True
        for bomb_pos, timer in extra_bombs:
            if self._bomb_hits(bomb_pos, pos) and timer - dt == 0:
                return True
        return False

    def _bomb_hits(self, bomb_pos: Coord, pos: Coord) -> bool:
        if max(abs(bomb_pos[0] - pos[0]), abs(bomb_pos[1] - pos[1])) > BLAST_RADIUS:
            return False
        return self._los_clear(bomb_pos, pos)

    def _los_clear(self, start: Coord, end: Coord) -> bool:
        if start == end:
            return True
        path = self._supercover_line(start, end)
        for idx in range(len(path) - 1):
            cx, cy = path[idx]
            nx, ny = path[idx + 1]
            dx = nx - cx
            dy = ny - cy
            if dx != 0 and dy != 0:
                horiz0 = self._edge_between((cx, cy), (nx, cy))
                horiz1 = self._edge_between((nx, cy), (nx, ny))
                vert0 = self._edge_between((cx, cy), (cx, ny))
                vert1 = self._edge_between((cx, ny), (nx, ny))
                h_blocked = (horiz0 in self.walls) or (horiz1 in self.walls)
                v_blocked = (vert0 in self.walls) or (vert1 in self.walls)
                if h_blocked and v_blocked:
                    return False
            else:
                edge = self._edge_between((cx, cy), (nx, ny))
                if edge in self.walls:
                    return False
        return True

    @staticmethod
    def _supercover_line(start: Coord, end: Coord) -> list[Coord]:
        x0, y0 = start
        x1, y1 = end
        tiles = [(x0, y0)]
        dx = x1 - x0
        dy = y1 - y0
        nx = abs(dx)
        ny = abs(dy)
        sign_x = 1 if dx > 0 else -1 if dx < 0 else 0
        sign_y = 1 if dy > 0 else -1 if dy < 0 else 0
        px, py = x0, y0
        ix = iy = 0
        while ix < nx or iy < ny:
            if (1 + 2 * ix) * ny == (1 + 2 * iy) * nx:
                px += sign_x
                py += sign_y
                ix += 1
                iy += 1
            elif (1 + 2 * ix) * ny < (1 + 2 * iy) * nx:
                px += sign_x
                ix += 1
            else:
                py += sign_y
                iy += 1
            tiles.append((px, py))
        return tiles

    def _bomb_distance_score(self, pos: Coord, action: int) -> float:
        relevant = [
            max(abs(bomb_pos[0] - pos[0]), abs(bomb_pos[1] - pos[1]))
            for bomb_pos, bomb in self.bombs.items()
            if bomb.timer <= 2
        ]
        if not relevant:
            return 10.0
        move_bonus = 0.25 if action in (FORWARD, BACKWARD) else 0.0
        stay_penalty = -0.5 if action == STAY else 0.0
        return min(relevant) + move_bonus + stay_penalty

    def _has_unknown_neighbor(self, pos: Coord) -> bool:
        for dx, dy in DIRS:
            neighbor = (pos[0] + dx, pos[1] + dy)
            if self._in_bounds(neighbor) and neighbor not in self.seen_cells:
                edge = self._edge_between(pos, neighbor)
                if edge is not None and edge not in self.walls:
                    return True
        return False

    @staticmethod
    def _in_bounds(pos: Coord) -> bool:
        return 0 <= pos[0] < GRID_SIZE and 0 <= pos[1] < GRID_SIZE

    def _load_fixed_map(self) -> None:
        try:
            import fixed_map
        except Exception:
            return
        self.walls.update(self._normalize_edges(getattr(fixed_map, "FIXED_WALLS", ())))
        self.destructible_edges.update(
            self._normalize_edges(getattr(fixed_map, "FIXED_DESTRUCTIBLE_EDGES", ()))
        )
        for pos, kind in getattr(fixed_map, "FIXED_COLLECTIBLES", {}).items():
            try:
                coord = (int(pos[0]), int(pos[1]))
                kind = str(kind)
            except Exception:
                continue
            if kind not in COLLECTIBLE_VALUES:
                continue
            self.known_collectibles[coord] = TargetInfo(
                kind, COLLECTIBLE_VALUES[kind], 0
            )
            self.fixed_collectibles.add(coord)
        self.fixed_respawns.update(
            {
                (int(pos[0]), int(pos[1])): int(steps)
                for pos, steps in getattr(fixed_map, "FIXED_RESPAWN_STEPS", {}).items()
            }
        )
        self.fixed_base_locations.update(
            (int(pos[0]), int(pos[1]))
            for pos in getattr(fixed_map, "FIXED_BASE_LOCATIONS", ())
        )

    def _activate_fixed_bases(self, own_base: Coord, step: int) -> None:
        for base in self.fixed_base_locations:
            if base == own_base:
                self.ally_bases.add(base)
                self.enemy_bases.pop(base, None)
                self.dismissed_fixed_bases.add(base)

    def _suspected_enemy_bases(self) -> set[Coord]:
        return {
            base
            for base in self.fixed_base_locations
            if base not in self.ally_bases
            and base not in self.enemy_bases
            and base not in self.dismissed_fixed_bases
        }

    def _respawn_steps(self, pos: Coord) -> int:
        return max(5, int(self.fixed_respawns.get(pos, RESPAWN_STEPS)))

    @staticmethod
    def _combat_target_value(kind: str, step: int) -> float:
        if kind == "enemy_base":
            return 12.0 if step < BASE_RUSH_STEP else 60.0
        if kind == "suspected_enemy_base":
            return 8.0 if step < BASE_RUSH_STEP else 36.0
        if kind == "enemy":
            return 10.0 if step < BASE_RUSH_STEP else 18.0
        return 0.0

    @staticmethod
    def _normalize_edges(edges: Iterable[object]) -> set[Edge]:
        normalized: set[Edge] = set()
        for edge in edges:
            try:
                a, b = edge  # type: ignore[misc]
                ca = (int(a[0]), int(a[1]))
                cb = (int(b[0]), int(b[1]))
            except Exception:
                continue
            normalized.add((ca, cb) if ca <= cb else (cb, ca))
        return normalized

    # ------------------------------------------------------------------
    # Generic helpers

    @staticmethod
    def _cell_is_visible(cell: np.ndarray) -> bool:
        return bool(cell[CH_VISIBLE] > 0.5 or np.any(cell[1:25] > 0.5))

    @staticmethod
    def _read_array(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        try:
            arr = np.asarray(value, dtype=np.float32)
            return arr if arr.size else None
        except Exception:
            return None

    @staticmethod
    def _read_location(
        observation: dict[str, Any], key: str = "location"
    ) -> Coord | None:
        loc = observation.get(key)
        if loc is None:
            return None
        try:
            return int(loc[0]), int(loc[1])
        except Exception:
            return None

    @staticmethod
    def _read_int(value: Any, default: int) -> int:
        try:
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            return int(value)
        except Exception:
            return default

    @staticmethod
    def _read_float(value: Any, default: float) -> float:
        try:
            if isinstance(value, (list, tuple)) and value:
                value = value[0]
            return float(value)
        except Exception:
            return default

    @staticmethod
    def _read_action_mask(observation: dict[str, Any]) -> list[int]:
        mask = observation.get("action_mask")
        if mask is None:
            return [1] * NUM_ACTIONS
        try:
            values = [1 if int(v) else 0 for v in mask]
            if len(values) < NUM_ACTIONS:
                values.extend([0] * (NUM_ACTIONS - len(values)))
            return values[:NUM_ACTIONS]
        except Exception:
            return [1] * NUM_ACTIONS

    @staticmethod
    def _is_frozen(observation: dict[str, Any], action_mask: list[int]) -> bool:
        frozen_ticks = AEManager._read_int(observation.get("frozen_ticks"), 0)
        if frozen_ticks > 0:
            return True
        return sum(action_mask) == 1 and action_mask[STAY] == 1

    @staticmethod
    def _legal(action_mask: list[int], action: int) -> bool:
        return 0 <= action < len(action_mask) and bool(action_mask[action])

    @staticmethod
    def _first_legal(preferences: tuple[int, ...], action_mask: list[int]) -> int:
        for action in preferences:
            if AEManager._legal(action_mask, action):
                return action
        return STAY
