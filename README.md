# Flappy Bird with Deep Q-Learning

This project trains an agent to play Flappy Bird using a Deep Q-Network (DQN), PyTorch, and the `FlappyBird-v0` Gymnasium environment.

## Requirements

- Python
- The packages listed in `requirements.txt`

## Setup

From the project directory, install the dependencies:

```bash
python -m pip install -r requirements.txt
```

## Train the agent

Start training with the `flappybirdv0` parameter set:

```bash
python agent.py flappybirdv0 --train
```

Training runs continuously. Press `Ctrl+C` to stop it. The best model is saved to `runs/flappybirdv0.pt`, and reward updates are appended to `runs/flappybirdv0.log`.

## Run the trained agent

To load the saved model and watch it play:

```bash
python agent.py flappybirdv0
```

The model file must exist at `runs/flappybirdv0.pt` before running in evaluation mode.

## Project files

- `agent.py` — training and evaluation loop, including DQN optimization.
- `dqn.py` — neural network used as the Q-function.
- `experience_replay.py` — experience replay buffer.
- `parameters.yaml` — hyperparameters for each parameter set.
- `requirements.txt` — Python dependencies.
- `runs/` — saved model checkpoints and training logs.

## Hyperparameters

The `flappybirdv0` configuration in `parameters.yaml` defines the environment and training settings, including the learning rate (`alpha`), discount factor (`gamma`), exploration schedule (`epsilon_*`), replay batch size, and target-network synchronization rate.
