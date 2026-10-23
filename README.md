# DQN, PPO and SAC vs. A*: navigation on a low-cost robot

Code, models and logs for a comparison of three deep reinforcement learning methods (DQN, PPO, SAC) and an A* planner on a 3D-printed differential-drive robot (Raspberry Pi 5, three ultrasonic sensors). The task is reaching a goal in a 0.91 m x 1.83 m corridor with 0, 1 or 2 obstacles.

## What was run

- **Simulation:** DQN, PPO and SAC were trained once per obstacle density in a custom Gymnasium environment. SAC uses a continuous-action version of it (`nav_env_continuous.py`). Each saved model was then evaluated on 100 held-out episodes (seeds 10000-10099).
- **Hardware:** DQN and A* ran on the robot, 3 trials per method per density (18 trials). PPO and SAC were not run on hardware.
- **A* baseline:** it was given no obstacle positions and planned once at startup. On hardware it avoided obstacles only through the same reactive safety override that DQN used.

## Results

Held-out simulation success (100 episodes each, 0 / 1 / 2 obstacles):

| Method | 0 | 1 | 2 |
|---|---|---|---|
| DQN | 99% | 51% | 43% |
| SAC | 95% | 51% | 40% |
| PPO (sampled actions with obstacles) | 100% | 49% | 42% |

PPO with deterministic actions scored 0% and 21% at 1 and 2 obstacles, mostly by timing out.

Hardware (3 trials per condition): both methods reached the goal by odometry in all 18 trials, and both touched an obstacle in every trial that had one. A* used fewer movement commands than DQN at every density and had higher odometry-estimated path efficiency with obstacles. Three trials per condition is too few for statistical testing.

## Known limitations

Each method was trained once with no fixed seed. Odometry was not independently calibrated, and the simulated turn angle and forward step are smaller than on the robot. Details are in the paper.

## Folders

- `simulation/` - environments (`nav_env.py`, `nav_env_continuous.py`), agents and training scripts, the A* planner, and `eval_sim.py`
- `hardware/` - Pi deployment code (`deploy.py`, `deploy_astar.py`), `odometry.py`, and `calibration/` test scripts
- `models/` - trained DQN (`.pt`), PPO and SAC (`.zip`) models
- `logs/` - terminal logs from the 18 hardware trials
- `analysis/` - log-parsing scripts and result CSVs

## Reproducing

Requires Python, PyTorch, Stable-Baselines3 2.9.0 and Gymnasium; the Pi code also needs `gpiozero` and `lgpio`.

- Train: set `NUM_OBSTACLES` and `SAVE_PATH` at the top of `train_corridor.py`, `train_ppo.py` or `train_sac.py`, then run it.
- Evaluate: `python eval_sim.py` (add `--algos PPO --sample-actions PPO` to sample PPO's actions).
- Hardware: set the model path and goal in `deploy.py` or `deploy_astar.py`, then run it on the Pi.

## AI assistance

Claude (Anthropic) was used to write and debug code.
