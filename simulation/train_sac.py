"""
train_sac.py

Trains a SAC agent on a continuous-action variant of the corridor
environment (nav_env_continuous.py), using stable-baselines3's standard
SAC implementation.

SAC requires a continuous action space -- confirmed incompatible with
the Discrete(4) space used by DQN/PPO/A* via direct testing. This uses
a genuinely different action representation: forward speed + turn rate,
applied simultaneously (see nav_env_continuous.py for full details) --
document this as an intentional scope decision in Methods, not silently.

This is a SIMULATION-ONLY comparison (no real hardware deployment),
matching the scope decision already made for PPO.

Install stable-baselines3 first if you don't have it:
    pip install stable-baselines3 --break-system-packages

Usage:
    python train_sac.py
"""

import time

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback

from nav_env_continuous import NavEnvContinuous


# ------------------------------------------------------------------ #
# Same corridor dimensions and robot/obstacle parameters as
# train_corridor.py / train_ppo.py, for a fair, comparable result.
# ------------------------------------------------------------------ #
ARENA_WIDTH_M = 1.83
ARENA_LENGTH_M = 0.91
ROBOT_RADIUS_M = 0.21
OBSTACLE_RADIUS_M = 0.10

# SET THIS to match whichever condition you're comparing against:
#   0 obstacles: NUM_OBSTACLES = 0, SAVE_PATH = "sac_corridor.zip"
#   1 obstacle:  NUM_OBSTACLES = 1, SAVE_PATH = "sac_1obs.zip"
#   2 obstacles: NUM_OBSTACLES = 2, SAVE_PATH = "sac_2obs.zip"
NUM_OBSTACLES = 2
SAVE_PATH = "sac_2obs.zip"

GOAL_RADIUS_M = 0.25
MAX_SENSOR_RANGE_M = 2.0
MOVE_SPEED_M = 0.10
TURN_SPEED_RAD = 0.30

TOTAL_TIMESTEPS = 150_000  # same budget as train_ppo.py, for comparability


class PrintProgressCallback(BaseCallback):
    """Same save-best-checkpoint + periodic printout logic as
    train_ppo.py -- tracks a rolling 100-episode success rate and saves
    the model whenever a new best is reached, rather than only saving
    whatever the final episode's weights happen to be. Included from
    the start this time, having already seen how much it mattered for
    PPO (38% vs. 56% final result, purely from where training ended)."""

    def __init__(self, save_path, print_every=20, best_window=100, verbose=0):
        super().__init__(verbose)
        self.save_path = save_path
        self.print_every = print_every
        self.best_window = best_window
        self.episode_outcomes = []
        self.start_time = time.time()
        self.best_success_rate = -1.0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        new_outcome_this_step = False
        for info, done in zip(infos, dones):
            if not done:
                continue
            if info.get("reached_goal"):
                self.episode_outcomes.append("goal")
            elif info.get("collided"):
                self.episode_outcomes.append("collision")
            else:
                self.episode_outcomes.append("timeout")
            new_outcome_this_step = True

        if new_outcome_this_step and len(self.episode_outcomes) >= self.best_window:
            recent_window = self.episode_outcomes[-self.best_window:]
            current_success_rate = recent_window.count("goal") / self.best_window
            if current_success_rate > self.best_success_rate:
                self.best_success_rate = current_success_rate
                self.model.save(self.save_path)

        if new_outcome_this_step and len(self.episode_outcomes) % self.print_every == 0:
            recent = self.episode_outcomes[-self.print_every:]
            success_rate = recent.count("goal") / len(recent)
            collision_rate = recent.count("collision") / len(recent)
            elapsed = time.time() - self.start_time
            print(
                f"Episode {len(self.episode_outcomes):5d} | "
                f"success: {success_rate:5.1%} | "
                f"collision: {collision_rate:5.1%} | "
                f"best (window={self.best_window}): {self.best_success_rate:5.1%} | "
                f"elapsed: {elapsed:6.1f}s"
            )
        return True


def main():
    print(f"Corridor arena: {ARENA_WIDTH_M}m x {ARENA_LENGTH_M}m")
    print(f"Obstacles: {NUM_OBSTACLES} | Save path: {SAVE_PATH}")
    print("Action space: continuous (forward speed + turn rate)")

    env = NavEnvContinuous(
        arena_width=ARENA_WIDTH_M,
        arena_length=ARENA_LENGTH_M,
        num_obstacles=NUM_OBSTACLES,
        obstacle_radius=OBSTACLE_RADIUS_M,
        robot_radius=ROBOT_RADIUS_M,
        goal_radius=GOAL_RADIUS_M,
        max_sensor_range=MAX_SENSOR_RANGE_M,
        move_speed=MOVE_SPEED_M,
        turn_speed=TURN_SPEED_RAD,
        max_steps=150,
    )

    model = SAC("MlpPolicy", env, verbose=0, device="cpu")
    callback = PrintProgressCallback(save_path=SAVE_PATH, print_every=20, best_window=100)

    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callback)

    print(f"\nTraining complete. Best checkpoint (success rate {callback.best_success_rate:.1%} over a {callback.best_window}-episode window) saved to {SAVE_PATH}")


if __name__ == "__main__":
    main()
