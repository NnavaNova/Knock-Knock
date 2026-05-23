"""Manages the AE model — scripted Bomberman agent.

The challenge environment (til_environment.bomberman_env) is a multi-agent
grid game with Bomberman mechanics: agents move on a 16x16 grid, collect
tiles for points, place bombs, and try to destroy the enemy team's base.

Strategy is a priority cascade re-evaluated every step:

  1. Survival.  If a bomb's blast radius reaches us in 2 ticks, flee.
  2. Combat.    If an enemy or enemy base sits where our bomb would hit,
                plant it (we'll flee on the next tick).
  3. Greedy.    Walk toward the closest visible mission tile, then recon,
                then resource — ordered by reward (+5, +1, +2 respectively;
                mission first because the EV per step is highest).
  4. Pressure.  If the enemy base is visible, head toward it.
  5. Explore.   Forward when possible, turn when blocked, periodic random
                turn to find new sectors.

An anti-stuck guard rotates when location doesn't change for 3+ steps.
We respect `action_mask` when the environment provides one.

Why scripted instead of RL: training a competitive Bomberman policy
needs ~100M+ env steps (days on a GPU). A well-shaped heuristic beats
the action=0 baseline by a huge margin and reaches a reasonable score
within minutes of work. RL is the right escalation if score plateaus.
"""

from __future__ import annotations

from typing import Any

import numpy as np


# Action indices, matching til_environment.actions.Action.
FORWARD = 0
BACKWARD = 1
LEFT = 2       # rotate 90 CCW (no movement)
RIGHT = 3      # rotate 90 CW (no movement)
STAY = 4
PLACE_BOMB = 5
NUM_ACTIONS = 6

# Direction encoding (matches the env): 0=RIGHT, 1=DOWN, 2=LEFT, 3=UP.
# (dx, dy) for moving one cell forward when facing each direction.
DIR_DX = (1, 0, -1, 0)
DIR_DY = (0, 1, 0, -1)

# Viewcone channel indices (from til_environment.observation.ViewChannel).
CH_VISIBLE = 0
CH_WALL_R, CH_WALL_D, CH_WALL_L, CH_WALL_U = 1, 2, 3, 4
CH_TILE_EMPTY, CH_TILE_RECON, CH_TILE_MISSION, CH_TILE_RESOURCE = 5, 6, 7, 8
CH_ALLY_AGENT, CH_ENEMY_AGENT, CH_ALLY_BASE, CH_ENEMY_BASE = 9, 10, 11, 12
CH_DWALL_R, CH_DWALL_D, CH_DWALL_L, CH_DWALL_U = 13, 14, 15, 16
CH_ALLY_BOMB, CH_ENEMY_BOMB = 17, 18
CH_ALLY_BOMB_TIMER, CH_ENEMY_BOMB_TIMER = 19, 20
# 21-24 are HP ratios, unused by the current strategy.

# Standard agent_viewcone layout per spec: (7 rows × 5 cols × 25 channels),
# agent at row 2, col 2. Rows are oriented relative to facing direction:
# +row = ahead, -row = behind; +col = right of agent, -col = left.
AGENT_ROW = 2
AGENT_COL = 2

# Tile collection priorities, by approximate value-per-step.
COLLECTIBLE_CHANNELS = (
    (CH_TILE_MISSION, 5.0),
    (CH_TILE_RESOURCE, 2.0),
    (CH_TILE_RECON, 1.0),
)

# Bomb's default blast radius from the upstream config.
BLAST_RADIUS = 2

# How many ticks of bomb timer constitutes "imminent" danger.
IMMINENT_BOMB_TICKS = 2

# How many same-location steps before we forcibly rotate.
STUCK_LIMIT = 3


class AEManager:
    def __init__(self):
        self._previous_location: tuple[int, int] | None = None
        self._stuck_steps: int = 0
        self._just_placed_bomb: bool = False
        self._explore_turn_counter: int = 0

    def ae(self, observation: dict[str, Any]) -> int:
        """Choose the next action given the current observation."""
        viewcone = self._read_viewcone(observation)
        direction = int(observation.get("direction", 0))
        location = self._read_location(observation)
        step = int(observation.get("step", 0))
        action_mask = self._read_action_mask(observation, default_length=NUM_ACTIONS)
        team_bombs = int(observation.get("team_bombs", 1))

        # Round boundary: the server calls /reset on step 0, but be defensive.
        if step == 0:
            self._previous_location = None
            self._stuck_steps = 0
            self._just_placed_bomb = False
            self._explore_turn_counter = 0

        scene = self._scan_viewcone(viewcone)
        action = self._decide(scene, direction, team_bombs, step, action_mask)

        # Anti-stuck guard. If we haven't moved (and we picked a movement
        # action), force a rotation to break the loop.
        if location is not None and location == self._previous_location:
            self._stuck_steps += 1
        else:
            self._stuck_steps = 0
        if self._stuck_steps >= STUCK_LIMIT:
            action = self._first_legal(
                (RIGHT, LEFT, BACKWARD, FORWARD, STAY), action_mask
            )
            self._stuck_steps = 0

        self._previous_location = location
        self._just_placed_bomb = action == PLACE_BOMB
        return int(action)

    # -------------------------------------------------- observation parsing

    @staticmethod
    def _read_viewcone(observation: dict[str, Any]) -> np.ndarray | None:
        """Return the agent's viewcone as a numpy array, or None.

        The wiki spec (and the upstream env) calls this 'agent_viewcone' and
        ships a (7, 5, 25) float array. The local README shows an older
        'viewcone' field with a 2D int layout. We accept either; downstream
        code adapts to whichever shape we get.
        """
        for key in ("agent_viewcone", "viewcone"):
            value = observation.get(key)
            if value is None:
                continue
            try:
                arr = np.asarray(value, dtype=np.float32)
                if arr.size:
                    return arr
            except Exception:
                continue
        return None

    @staticmethod
    def _read_location(observation: dict[str, Any]) -> tuple[int, int] | None:
        loc = observation.get("location")
        if loc is None:
            return None
        try:
            return (int(loc[0]), int(loc[1]))
        except Exception:
            return None

    @staticmethod
    def _read_action_mask(
        observation: dict[str, Any], default_length: int
    ) -> list[int]:
        mask = observation.get("action_mask")
        if mask is None:
            return [1] * default_length
        try:
            return [1 if int(v) else 0 for v in mask]
        except Exception:
            return [1] * default_length

    # ----------------------------------------------------- viewcone scanning

    def _scan_viewcone(self, view: np.ndarray | None) -> dict[str, Any]:
        """Extract decision-relevant features from the viewcone.

        Returned dict keys:
          'bomb_threats' — list of (delta_row, delta_col, timer) for bombs
                           that could hit the agent's cell soon.
          'enemy_in_blast' — bool, True if an enemy or enemy base sits within
                             our hypothetical bomb's blast radius.
          'forward_blocked' — bool, True if a wall blocks our next forward step.
          'targets' — list of (priority, delta_row, delta_col) for visible
                       collectibles / enemy base, lower priority value = better.
        """
        scene: dict[str, Any] = {
            "bomb_threats": [],
            "enemy_in_blast": False,
            "forward_blocked": True,  # default safe to assume blocked
            "targets": [],
            "has_channels": False,
        }
        if view is None or view.size == 0:
            return scene

        # Three viewcone shape modes we tolerate:
        #   (R, C, 25) — full channel layout (preferred)
        #   (R, C)     — simplified, single int per cell (legacy)
        #   (25, R, C) — defensive in case channels are first axis
        if view.ndim == 3 and view.shape[-1] == 25:
            view3 = view
        elif view.ndim == 3 and view.shape[0] == 25 and view.shape[-1] != 25:
            view3 = np.transpose(view, (1, 2, 0))
        elif view.ndim == 2:
            return self._scan_2d_view(view, scene)
        else:
            return scene

        scene["has_channels"] = True
        rows, cols, _ = view3.shape
        ar = min(AGENT_ROW, rows - 1) if rows > 0 else 0
        ac = min(AGENT_COL, cols - 1) if cols > 0 else 0

        # Forward blocked: wall on the U/back edge of cell ahead, or wall on
        # our own forward edge, or any destructible-wall edge present.
        # "Ahead" in the viewcone is the row immediately greater than ar.
        forward_row = ar + 1
        if forward_row < rows:
            here = view3[ar, ac]
            ahead = view3[forward_row, ac]
            forward_wall = (
                here[CH_WALL_D] > 0.5
                or here[CH_WALL_R] > 0.5   # see note below
                or ahead[CH_WALL_U] > 0.5
                or ahead[CH_WALL_L] > 0.5
                or here[CH_DWALL_D] > 0.5
                or ahead[CH_DWALL_U] > 0.5
                or ahead[CH_ALLY_AGENT] > 0.5  # ally also blocks
            )
            scene["forward_blocked"] = bool(forward_wall)
        else:
            scene["forward_blocked"] = True

        # Bomb threats: any bomb close enough that its blast reaches us soon.
        ally_bomb = view3[..., CH_ALLY_BOMB]
        enemy_bomb = view3[..., CH_ENEMY_BOMB]
        bomb_mask = (ally_bomb > 0.5) | (enemy_bomb > 0.5)
        if bomb_mask.any():
            timers = np.maximum(
                view3[..., CH_ALLY_BOMB_TIMER],
                view3[..., CH_ENEMY_BOMB_TIMER],
            )
            for (r, c) in np.argwhere(bomb_mask).tolist():
                dr = r - ar
                dc = c - ac
                # Bombs blast in straight lines; treat 'reach us' as either
                # same row OR same column within BLAST_RADIUS.
                blast_can_reach = (dr == 0 and abs(dc) <= BLAST_RADIUS) or (
                    dc == 0 and abs(dr) <= BLAST_RADIUS
                )
                t = float(timers[r, c])
                # Treat 0/missing timer as "about to detonate" for safety.
                if blast_can_reach and (t <= IMMINENT_BOMB_TICKS or t == 0):
                    scene["bomb_threats"].append((dr, dc, t))

        # Enemy or enemy base in our own potential blast (we're at (ar, ac),
        # blast goes ±BLAST_RADIUS along both axes from us).
        enemy_grid = view3[..., CH_ENEMY_AGENT] + view3[..., CH_ENEMY_BASE]
        for (r, c) in np.argwhere(enemy_grid > 0.5).tolist():
            dr = r - ar
            dc = c - ac
            in_blast = (dr == 0 and abs(dc) <= BLAST_RADIUS) or (
                dc == 0 and abs(dr) <= BLAST_RADIUS
            )
            if in_blast:
                scene["enemy_in_blast"] = True
                break

        # Targets, ordered: enemy_base (highest score event), then
        # collectibles by EV. Nearest within each category wins.
        targets: list[tuple[int, int, int]] = []

        def _add_nearest(channel: int, priority: int) -> None:
            cells = np.argwhere(view3[..., channel] > 0.5)
            if len(cells) == 0:
                return
            best = min(cells.tolist(), key=lambda rc: abs(rc[0] - ar) + abs(rc[1] - ac))
            targets.append((priority, best[0] - ar, best[1] - ac))

        _add_nearest(CH_ENEMY_BASE, 0)
        for prio, (channel, _value) in enumerate(COLLECTIBLE_CHANNELS, start=1):
            _add_nearest(channel, prio)
        # Enemy agent — chase if we want to bomb it; lower priority than
        # collectibles, higher than nothing.
        _add_nearest(CH_ENEMY_AGENT, len(COLLECTIBLE_CHANNELS) + 1)

        scene["targets"] = sorted(targets, key=lambda t: (t[0], abs(t[1]) + abs(t[2])))
        return scene

    @staticmethod
    def _scan_2d_view(view: np.ndarray, scene: dict[str, Any]) -> dict[str, Any]:
        """Fallback for the legacy 2D-int viewcone format.

        We don't know the exact encoding without testing locally, so we use
        coarse heuristics: any non-zero cell ahead is "blocked", and any
        non-zero cell in view is a vague navigation target.
        """
        rows, cols = view.shape
        ar = min(AGENT_ROW, rows - 1) if rows > 0 else 0
        ac = min(AGENT_COL, cols - 1) if cols > 0 else 0
        forward_row = ar + 1
        if forward_row < rows:
            scene["forward_blocked"] = bool(view[forward_row, ac] != 0)
        # Any visible non-zero cell becomes a single low-priority target so
        # we at least drift toward activity instead of standing still.
        nonzero = np.argwhere(view != 0)
        if len(nonzero) > 0:
            best = min(nonzero.tolist(), key=lambda rc: abs(rc[0] - ar) + abs(rc[1] - ac))
            scene["targets"] = [(99, best[0] - ar, best[1] - ac)]
        return scene

    # ---------------------------------------------------------- decision

    def _decide(
        self,
        scene: dict[str, Any],
        direction: int,
        team_bombs: int,
        step: int,
        action_mask: list[int],
    ) -> int:
        # 1) SURVIVAL — if a bomb will hit us soon, get out of its line.
        if scene["bomb_threats"]:
            return self._flee_action(scene, action_mask)

        # 2) COMBAT — enemy or enemy base in our blast radius, and we have
        #    bombs. Place the bomb and let the next-tick survival branch
        #    pull us out.
        if (
            scene["enemy_in_blast"]
            and team_bombs > 0
            and action_mask[PLACE_BOMB]
            and not self._just_placed_bomb
        ):
            return PLACE_BOMB

        # 3) NAVIGATE toward best target if any are visible.
        if scene["targets"]:
            _, dr, dc = scene["targets"][0]
            move = self._move_toward(dr, dc, scene["forward_blocked"], action_mask)
            if move is not None:
                return move

        # 4) EXPLORE — bias toward FORWARD with periodic turns so we don't
        #    plough straight into a corner forever.
        if not scene["forward_blocked"] and action_mask[FORWARD]:
            self._explore_turn_counter += 1
            # Inject a turn every ~8 forward steps to discover new areas.
            if self._explore_turn_counter >= 8:
                self._explore_turn_counter = 0
                turn = RIGHT if (step // 8) % 2 == 0 else LEFT
                if action_mask[turn]:
                    return turn
            return FORWARD

        # Forward is blocked: rotate to find a new heading.
        rotate = RIGHT if step % 2 == 0 else LEFT
        if action_mask[rotate]:
            return rotate
        return self._first_legal(
            (LEFT, RIGHT, BACKWARD, STAY, FORWARD), action_mask
        )

    # ---------------------------------------------------- action selection

    @staticmethod
    def _flee_action(scene: dict[str, Any], action_mask: list[int]) -> int:
        """Pick an action that takes us out of the closest bomb's blast line."""
        # Bomb threats are sorted-ish by manhattan distance; use the closest.
        threats = sorted(scene["bomb_threats"], key=lambda t: abs(t[0]) + abs(t[1]))
        dr, dc, _t = threats[0]
        # If bomb shares our row (dr == 0): move forward or backward away.
        if dr == 0:
            # Bomb to our right (dc>0) or left (dc<0). Move opposite if not blocked.
            # Forward is +row, Backward is -row — neither directly moves sideways,
            # but stepping perpendicular leaves the row. Prefer forward unless blocked.
            if not scene["forward_blocked"] and action_mask[FORWARD]:
                return FORWARD
            if action_mask[BACKWARD]:
                return BACKWARD
            # Last resort: turn so a subsequent step takes us off the line.
            return AEManager._first_legal((RIGHT, LEFT, STAY), action_mask)
        # If bomb is ahead (dr>0): retreat.
        if dr > 0:
            if action_mask[BACKWARD]:
                return BACKWARD
            return AEManager._first_legal((RIGHT, LEFT, STAY), action_mask)
        # Bomb is behind (dr<0): push forward away from it.
        if not scene["forward_blocked"] and action_mask[FORWARD]:
            return FORWARD
        return AEManager._first_legal((LEFT, RIGHT, BACKWARD, STAY), action_mask)

    @staticmethod
    def _move_toward(
        dr: int, dc: int, forward_blocked: bool, action_mask: list[int]
    ) -> int | None:
        """Convert a viewcone-relative target offset into one action.

        dr: positive = target ahead, negative = behind.
        dc: positive = target right of us, negative = left.
        """
        if dr == 0 and dc == 0:
            return None  # already there

        # If clearly ahead and forward is open, step forward.
        if dr > 0 and not forward_blocked and action_mask[FORWARD]:
            return FORWARD

        # If clearly behind, back up.
        if dr < 0 and abs(dr) > abs(dc) and action_mask[BACKWARD]:
            return BACKWARD

        # Lateral target: turn toward it so the next tick can step forward.
        if dc > 0 and action_mask[RIGHT]:
            return RIGHT
        if dc < 0 and action_mask[LEFT]:
            return LEFT

        # Target ahead but blocked: try to turn around obstacle.
        if dr > 0 and forward_blocked:
            for option in (RIGHT, LEFT):
                if action_mask[option]:
                    return option

        return None

    @staticmethod
    def _first_legal(preferences: tuple[int, ...], action_mask: list[int]) -> int:
        for action in preferences:
            if 0 <= action < len(action_mask) and action_mask[action]:
                return action
        # Should never fire — STAY is always legal in a well-formed env.
        return STAY
