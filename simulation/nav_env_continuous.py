"""
nav_env_continuous.py

Continuous-action variant of NavEnv, built specifically for SAC (which
requires a continuous action space -- confirmed incompatible with the
Discrete(4) action space used by DQN/PPO/A* via direct testing:
stable-baselines3's SAC raises an explicit error on a Discrete action
space).

Subclasses NavEnv directly and reuses its sensor simulation, collision
checking, observation vector, reset logic, and terminal rewards
UNCHANGED -- only the action space and per-step motion/reward-shaping
are overridden, so results stay as comparable to DQN/PPO/A* as a
genuinely different action representation allows.

Action space: Box(low=-1, high=1, shape=(2,))
    action[0]: forward speed. Rescaled internally from [-1,1] to [0,1]
               (no backward motion, matching the discrete env's
               forward-only action).
    action[1]: turn rate, [-1,1]. Positive = turn left, negative = turn
               right.

IMPORTANT DIFFERENCE from the discrete environment: forward motion and
turning are applied SIMULTANEOUSLY here (standard continuous
differential-drive control), whereas the discrete env's actions are
mutually exclusive per step (either move straight OR turn in place,
never both). This lets the continuous agent arc through turns rather
than stopping to pivot. This is a genuine, intentional difference in
the action representation -- document it as such in Methods, not as an
oversight.
"""

import numpy as np
from gymnasium import spaces

from nav_env import NavEnv


class NavEnvContinuous(NavEnv):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

    def step(self, action):
        self.steps += 1

        action = np.clip(action, -1.0, 1.0)
        forward_frac = (action[0] + 1.0) / 2.0  # rescale [-1,1] -> [0,1], no backward motion
        turn_rate = float(action[1])  # already [-1, 1]

        # forward and turning applied simultaneously (unlike the discrete
        # env's mutually-exclusive actions) -- standard continuous
        # differential-drive control, allows arcing motion
        new_pos = self.robot_pos + (forward_frac * self.move_speed) * np.array(
            [np.cos(self.robot_heading), np.sin(self.robot_heading)]
        )
        self.robot_pos = new_pos
        self.robot_heading += turn_rate * self.turn_speed

        self.robot_heading = self._wrap_angle(self.robot_heading)
        self.robot_pos = np.clip(
            self.robot_pos, 0.0, [self.arena_width, self.arena_length]
        )

        dist_to_goal = np.linalg.norm(self.goal_pos - self.robot_pos)
        collided = self._check_collision()
        reached_goal = dist_to_goal < self.goal_radius

        reward = self._compute_reward_continuous(dist_to_goal, collided, reached_goal, forward_frac, turn_rate)
        self._prev_dist_to_goal = dist_to_goal

        terminated = bool(collided or reached_goal)
        truncated = bool(self.steps >= self.max_steps)

        obs = self._get_obs()
        info = {
            "collided": collided,
            "reached_goal": reached_goal,
            "dist_to_goal": dist_to_goal,
        }
        return obs, reward, terminated, truncated, info

    def _compute_reward_continuous(self, dist_to_goal, collided, reached_goal, forward_frac, turn_rate):
        """Continuous-action analog of NavEnv._compute_reward -- same
        core progress-based reward and terminal rewards, but the
        forward-confidence bonus and turn penalty are scaled by action
        MAGNITUDE instead of applied as fixed per-discrete-action
        amounts, since continuous actions aren't all-or-nothing."""
        if reached_goal:
            return 100.0
        if collided:
            return -100.0

        progress = self._prev_dist_to_goal - dist_to_goal
        reward = progress * 10.0
        reward -= 0.05  # per-step penalty, same as discrete env

        # continuous analog of the discrete env's forward-confidence
        # bonus: scale by how strongly forward motion was chosen
        front_reading = self._simulate_sensor(self._sensor_angles[0])
        if front_reading > 0.3:
            reward += 0.10 * forward_frac

        # continuous analog of the discrete env's turn penalty: scale
        # by how much turning was applied
        reward -= 0.035 * abs(turn_rate)

        return reward
