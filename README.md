# Reinforcement Learning for Autonomous Tactical Behavior
## Multi-Domain (Air-Ground-Sea) Constructive Simulation
### Towards Digital Twin-Enabled Training Systems

> Phase 1: 2D Air Domain -- Single Agent RL

---

## Project Structure

| Directory        | Type        | Purpose                                          |
|------------------|-------------|--------------------------------------------------|
| envs/            | Python pkg  | Gymnasium environment (flight_env, world, etc.)  |
| physics/         | Python pkg  | Aircraft dynamics (RK4, forces, state)           |
| observation/     | Python pkg  | State vector construction and normalization      |
| reward/          | Python pkg  | Pluggable reward components (ablation-ready)     |
| training/        | Python pkg  | Training scripts (PPO, DQN, SAC)                 |
| evaluation/      | Python pkg  | Benchmarking, metrics, report generation         |
| baselines/       | Python pkg  | Random, Greedy, PID, DQN baseline agents         |
| curriculum/      | Python pkg  | Curriculum learning scheduler                    |
| agents/          | Python pkg  | SB3 agent wrappers and configs                   |
| rendering/       | Python pkg  | Pygame renderer and tactical HUD                 |
| replay/          | Python pkg  | Episode recorder and playback system             |
| metrics/         | Python pkg  | Research metrics tracker and comparison tables   |
| experiments/     | Python pkg  | Experiment manager (per-run folders)             |
| utils/           | Python pkg  | Shared helpers (config, logging, geometry)       |
| configs/         | YAML        | All hyperparameters -- zero magic numbers in code|
| data/            | Data        | Terrain, wind, scenarios, obstacle layouts       |
| assets/          | Static      | Images, icons, fonts for renderer                |
| logs/            | Runtime     | TensorBoard + training text logs                 |
| models/          | Runtime     | Saved checkpoints and best model weights         |
| tests/           | Tests       | Unit + integration tests (pytest)                |
| scripts/         | Entry pts   | train.py, evaluate.py, compare_baselines.py      |
| notebooks/       | Research    | Analysis and visualization notebooks             |
| docs/            | Docs        | Architecture documentation                       |

## Quickstart

    # 1. Install dependencies
    pip install -r requirements.txt

    # 2. Run unit tests (Stage 1 gate)
    pytest tests/ -v

    # 3. Manual keyboard control (after Stage 5)
    python scripts/manual_control.py

    # 4. Train PPO agent (after Stage 11)
    python scripts/train.py --agent ppo --config configs/ppo_phase1.yaml

    # 5. Compare all baselines (after Stage 15)
    python scripts/compare_baselines.py

    # 6. View TensorBoard
    tensorboard --logdir logs/tensorboard/

## Build Stages

    Stage 1  : Physics engine (RK4 dynamics)         [IN PROGRESS]
    Stage 2  : World + Observation builder
    Stage 3  : Gymnasium environment skeleton
    Stage 4  : Reward + Termination logic
    Stage 5  : Pygame renderer + tactical HUD
    Stage 6  : Episode replay system
    Stage 7  : Experiment manager
    Stage 8  : Research metrics tracker
    Stage 9  : Baseline agent suite (Random/Greedy/PID)
    Stage 10 : Config system (YAML-driven)
    Stage 11 : PPO training
    Stage 12 : DQN training
    Stage 13 : Curriculum learning
    Stage 14 : SAC + continuous action space
    Stage 15 : Full baseline comparison + report

## Roadmap

    Phase 1  (NOW)  : 2D Air Domain, Single Agent
    Phase 2         : 2D Multi-Domain (Air + Ground + Sea), MARL
    Phase 3         : 3D + Digital Twin Integration