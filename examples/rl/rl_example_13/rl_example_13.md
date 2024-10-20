# EXample 13: REINFORCE algorithm on CartPole

The DQN algorithm we used in examples <a href="../rl_example_12/rl_example_12.md">EXample 12: DQN algorithm on Gridworld</a>
and <a href="../rl_example_15/rl_example_15.md">EXample 15: DQN algorithm on Gridworld with experience replay</a> approximate a value function
and in particular the Q state-action value function. However, in Reinforcement Learning we are interested in policies since a policydictates how 
an agent behaves in a given state.


Policy-based methods on the other hand, approximate policies directly. 
In this example, we will use one of the earliest policy-based methods namely the
REINFORCE algorithm. In particular, we will train an agent that learns how to balance
a pole placed on a moving cart. In order to do this, we will use the ```gymnasium.CartPole``` environment.

We will use the policy network from the book <a href="https://www.manning.com/books/deep-reinforcement-learning-in-action">Deep Reinforcement Learning in Action</a>
by Manning Publications.

The REINFORCE algorithm is implemented in the class <a href="https://github.com/pockerman/cuberl/blob/master/include/cubeai/rl/algorithms/pg/simple_reinforce.h">ReinforceSolver</a>
The solver is passed to the ```RLSerialAgentTrainer``` class that manages the loop over the specified number of episodes.
The ```ReinforceSolver``` class overrides some virtual methods defined in the ```RLSolverBase``` class.


## The ```PolicyImpl```

In policy gradient methods, we estimate the policy function directly. We need therefore a way to
represent it. We will use PyTorch to model the policy network. We will also 






### Executing the policy

We saved the ```PolicyImpl``` model. We will load it on a Python script and run the environment in order to view how the agent is doing it.


## Summary

This example introduced the ```ReinforceSolver``` class that models the REINFORCE algorithm.

The RINFORCE algorithm, just like all policy gradient methods, suffers from high variance in the gradient estimation.
There are several reasons behind this high variance; e.g. sparse rewards or environment randomness.
Regardless of reasons behind high variance in the gradient, its effect can be detrimental during learning as it destabilizes it. Hence, reducing the high variance is important for feasible training. 
