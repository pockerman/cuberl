# EXample 12: DQN algorithm on Gridworld

In this example, we will train an agent so that it learns to navigate itself in a grid.
Specifically, we will be using the ```Gridworld``` environmant from the book <a href="https://www.manning.com/books/deep-reinforcement-learning-in-action">Deep Reinforcement Learning in Action</a>.
We have implemented this environment in <a herf="https://github.com/pockerman/rlenvs_from_cpp">rlenvs_from_cpp</a>; check the class <a href="https://github.com/pockerman/rlenvs_from_cpp/blob/master/src/rlenvs/envs/grid_world/grid_world_env.h">Gridworld</a>. 

We will use the DQN algorithm, see <a href="https://www.manning.com/books/deep-reinforcement-learning-in-action">Deep Reinforcement Learning in Action</a> and references therein,
in order to train our agent and we will TensorBoard to monitor the training. We will use a static environment configuration in this example something that makes this problem a lot easier to work on.

We will code the same model as is done in the <a href="https://www.manning.com/books/deep-reinforcement-learning-in-action">Deep Reinforcement Learning in Action</a> book so you may also want to follow the code therein.



