# This Neural network bot for Rocket League.
In order to use to use this bot you have to install [this plugin](https://bakkesplugins.com/) and python of course. And Epic Games version of the game is preferred.
This bot uses reinforcement learning and especcaly Proximal Policy Optimization (PPO) for training.

## Run
```python
python run.py
```
You are able to configure launch by changing the file rlbot.cfg.

## Training
```python
python ppo_training.py --tag "<tag for the run>" -n <a number of instances of the game>
```

All settings are in the file "ppocket_rocket\ppo_cfg.yaml"

Also this project consists the training with [simulator of rocket league](https://github.com/ZealanL/RocketSim). But it has bugs and it's not debbuged. assets.path, sim_requirements.txt, SDL2.dll, rlviser.exe, umodel.ext are files for the simulator. assets.path should consist a path to your folder with Rocket League.

dqn_training.py - is training by the algorithm DQN (Deep Q-Network). But it didn't work well and the code is not supported. I leave this code for the case if i will use an another off-policy method. 