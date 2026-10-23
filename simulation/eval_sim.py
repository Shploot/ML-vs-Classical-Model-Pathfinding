"""
eval_sim.py

Held-out evaluation of the saved simulation models (DQN, PPO, SAC) at
0, 1 and 2 obstacles.

Why this exists: the "best success" numbers printed during training are
the highest rolling 100-episode window seen during training, measured on
training episodes. The maximum of a noisy curve is biased upward. This
script loads each SAVED model and runs it on fresh, seeded episodes with
NO exploration, so the numbers are an unbiased estimate of what the
saved checkpoint actually does.

Design choices
- Every model sees the same episode seeds (SEED_BASE .. SEED_BASE+N-1).
  Training used unseeded resets, so these layouts are new, and the same
  layouts are shared across algorithms for a paired comparison.
- DQN: greedy action (epsilon = 0). PPO / SAC: deterministic=True.
- Each model runs in the environment it was trained in, with the same
  parameters as train_corridor.py / train_ppo.py / train_sac.py.
  PPO and DQN use the discrete NavEnv; SAC uses NavEnvContinuous.
- Reports success / collision / timeout rates with 95% Wilson intervals.

Put this file in the same folder as nav_env.py, nav_env_continuous.py
and dqn_agent.py (your simulation/ folder). Models are looked for in the
current folder, then ../models, or pass --model-dir.

Usage:
    python eval_sim.py
    python eval_sim.py --episodes 200
"""

import argparse
import csv
import math
import os
import sys

import numpy as np
import torch

from nav_env import NavEnv
from nav_env_continuous import NavEnvContinuous
from dqn_agent import DQNAgent

# ---- must match the training scripts ---------------------------------- #
ARENA_WIDTH_M = 1.83
ARENA_LENGTH_M = 0.91
ROBOT_RADIUS_M = 0.21
OBSTACLE_RADIUS_M = 0.10
GOAL_RADIUS_M = 0.25
MAX_SENSOR_RANGE_M = 2.0
MOVE_SPEED_M = 0.10
TURN_SPEED_RAD = 0.30
MAX_STEPS = 150

MODEL_FILES = {
    "DQN": {0: "nav_dqn_corridor.pt", 1: "nav_dqn_1obs.pt", 2: "nav_dqn_2obs.pt"},
    "PPO": {0: "ppo_corridor.zip", 1: "ppo_1obs.zip", 2: "ppo_2obs.zip"},
    "SAC": {0: "sac_corridor.zip", 1: "sac_1obs.zip", 2: "sac_2obs.zip"},
}


def make_env(algo, num_obstacles):
    cls = NavEnvContinuous if algo == "SAC" else NavEnv
    return cls(
        arena_width=ARENA_WIDTH_M,
        arena_length=ARENA_LENGTH_M,
        num_obstacles=num_obstacles,
        obstacle_radius=OBSTACLE_RADIUS_M,
        robot_radius=ROBOT_RADIUS_M,
        goal_radius=GOAL_RADIUS_M,
        max_sensor_range=MAX_SENSOR_RANGE_M,
        move_speed=MOVE_SPEED_M,
        turn_speed=TURN_SPEED_RAD,
        max_steps=MAX_STEPS,
    )


def find_model(filename, model_dir):
    candidates = []
    if model_dir:
        candidates.append(os.path.join(model_dir, filename))
    candidates += [filename, os.path.join("..", "models", filename)]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def load_policy(algo, path, env, sample=False):
    """Returns a function obs -> action.
    Default: exploration off (DQN greedy; PPO/SAC deterministic).
    sample=True (PPO/SAC only): draw actions from the policy's own
    distribution, the way the policy behaved during training."""
    if algo == "DQN":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        agent = DQNAgent(env.observation_space.shape[0], env.action_space.n, device)
        agent.load(path)
        return lambda obs: agent.select_action(obs, greedy=True)

    from stable_baselines3 import PPO, SAC
    cls = PPO if algo == "PPO" else SAC
    model = cls.load(path, env=env, device="cpu")

    def act(obs):
        action, _ = model.predict(obs, deterministic=not sample)
        return action

    return act


def run_episode(env, policy, seed):
    torch.manual_seed(seed)  # makes sampled-action runs reproducible
    obs, _ = env.reset(seed=seed)
    steps = 0
    while True:
        obs, _, terminated, truncated, info = env.step(policy(obs))
        steps += 1
        if terminated or truncated:
            break
    if info.get("reached_goal"):
        return "goal", steps
    if info.get("collided"):
        return "collision", steps
    return "timeout", steps


def wilson(k, n, z=1.96):
    """95% Wilson score interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--seed-base", type=int, default=10_000)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--out", default="eval_results.csv")
    parser.add_argument("--algos", nargs="*", default=None,
                        help="only evaluate these, e.g. --algos PPO")
    parser.add_argument("--obstacles", nargs="*", type=int, default=None,
                        help="only these obstacle counts, e.g. --obstacles 1 2")
    parser.add_argument("--sample-actions", nargs="*", default=[],
                        help="PPO/SAC: sample actions instead of deterministic, e.g. --sample-actions PPO")
    args = parser.parse_args()

    rows = []
    print(f"Held-out evaluation: {args.episodes} episodes per model, "
          f"seeds {args.seed_base}..{args.seed_base + args.episodes - 1}; action mode per row (see Mode column)\n")
    header = f"{'Algo':<5}{'Obs':>4}{'Mode':>15}{'Success':>9}{'95% CI':>16}{'Collide':>9}{'Timeout':>9}{'Steps(succ)':>13}"
    print(header)
    print("-" * len(header))

    sample_set = {a.upper() for a in args.sample_actions}
    if "DQN" in sample_set:
        print("Note: DQN has no sampling mode here (it is epsilon-greedy); ignoring DQN.")
        sample_set.discard("DQN")

    for algo, files in MODEL_FILES.items():
        if args.algos and algo not in {a.upper() for a in args.algos}:
            continue
        for n_obs, filename in files.items():
            if args.obstacles is not None and n_obs not in args.obstacles:
                continue
            path = find_model(filename, args.model_dir)
            if path is None:
                print(f"{algo:<5}{n_obs:>4}   (model '{filename}' not found -- skipped)")
                continue
            env = make_env(algo, n_obs)
            sampled = algo in sample_set
            mode = "sampled" if sampled else ("greedy" if algo == "DQN" else "deterministic")
            policy = load_policy(algo, path, env, sample=sampled)

            outcomes, success_steps = [], []
            for i in range(args.episodes):
                outcome, steps = run_episode(env, policy, args.seed_base + i)
                outcomes.append(outcome)
                if outcome == "goal":
                    success_steps.append(steps)

            n = len(outcomes)
            k = outcomes.count("goal")
            lo, hi = wilson(k, n)
            coll = outcomes.count("collision") / n
            tout = outcomes.count("timeout") / n
            mean_steps = float(np.mean(success_steps)) if success_steps else float("nan")
            print(f"{algo:<5}{n_obs:>4}{mode:>15}{k / n:>8.0%} {f'[{lo:.0%}, {hi:.0%}]':>16}"
                  f"{coll:>9.0%}{tout:>9.0%}{mean_steps:>13.1f}")
            rows.append({
                "algo": algo, "obstacles": n_obs, "mode": mode, "episodes": n,
                "success_rate": k / n, "ci_low": lo, "ci_high": hi,
                "collision_rate": coll, "timeout_rate": tout,
                "mean_steps_on_success": mean_steps, "model_file": path,
            })

    if rows:
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nWrote {args.out}")
    else:
        print("\nNo models found. Put the model files next to this script or in ../models.")
        sys.exit(1)


if __name__ == "__main__":
    main()
