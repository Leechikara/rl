# torch_rl : A starting Reinforcement Learning PyTorch Library

Torch_rl is a (small) Sequential Learning (and Reinforcement Learning) library for PyTorch that can be used to:
* learn policies over openAI Gym environments
* learn policies with DQN or Policy Gradient techniques (more to come)
* Model complex environements (not only reward-based environment)

Note that the library will evolve during the next months.

## Gerating Documentation

Go to the `docs` directory then: `PYTHONPATH=.. make html`. The generated HTML file are under `_build.html/index.html`

## Key packages

* the `torch_rl.core` package contains core classes. A classical RL environment is modeled using a triplet:
  * `World`: describes the physics of the world
  * `Sensor`: describes a (partial) view over the world.
  * `Task`: describes the task to solve. It can basically be defined through a `Reward` function
* the `Env` class is used to cast a triplet `(World,Sensor,Task)` to an openAI Gym Environment
* the `torch_rl.core.sensors` propose some basic tools for building high-level sensors from other sensors
* the `torch_rl.core.spaces` propose complex action and observation spaces

* the `torch_rl.policies` package contains policies definitions
  * `Policy` is the classical (non-learning) definition of a policy.

* the `torch_rl.learners` contains learning algorithms
  * DeepQN
  * PolicyGradient
  * BatchPolicyGradient (where multiple trajectories are sampled simultaneously)
  * RecurrentPolicyGradient
  * (...more to come...)

* the `tutorials` directory contains simple examples




