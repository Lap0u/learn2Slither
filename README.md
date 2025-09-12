# Project Title

Learn2Slither

## Description

A reinforcement learning environment for the snake game using pytorch and pygame.

## Installation

To set up the environment, run the following command:

```bash
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

## Norm

To check the norm run

```bash
pip install flake8
flake8 *.py src/*.py
```

## Usage

To start the training, run

````bash
```bash
python3 main.py --session 5000
````

This will start a training session of 5000 games. The model will be saved afterwards.

## Testing the model

```bash
python3 play.py --mode multiplay --num_games 5000 --path ./models/low_epsilon_3000_full_rew.pth
```

This will launch 5000 sessions of the game using the specified model.

## Visualizaing the model

```bash
python3 play.py --mode play --path ./models/low_epsilon_3000_full_rew.pth --speed 5
```

This will launch a game window using the specified model. The speed parameter controls the speed of the game. Step by step execution is possible by specifying --step-by-step in the command
