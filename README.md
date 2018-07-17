# Deep RL for Swarm Systems
This repository contains the codebase for the paper "Deep Reinforcement Learning for Swarm Systems". The code for TRPO is based on the OpenAI Baselines and the environment structure is inspired by OpenAI's multi-agent particle environments.

https://github.com/openai/baselines/

https://github.com/openai/multiagent-particle-envs/

## Installation
```
(optional) virtualenv -p python3 ~/virtual_envs/drl_for_swarms
(optional) source ~/virtual_envs/drl_for_swarms/bin/activate

git clone git@github.com:LCAS/deep_rl_for_swarms.git
cd deep_rl_for_swarms
pip install -e .
```

## Run
single core
```
cd deep_rl_for_swarms
python run_multiagent_trpo.py
```

multi core
```
cd deep_rl_for_swarms
mpirun -np 4 python run_multiagent_trpo.py
```