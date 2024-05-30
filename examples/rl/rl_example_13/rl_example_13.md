# EXample 13: REINFORCE algorithm on CartPole

In this example, we will train an agent so that it learns to 
balance the f....We will use one of the earliest policy gradient methods namely the
REINFORCE algorithm. 

We will use the ```gymnasium.CartPole``` environment.

## The ```PolicyImpl```

In policy gradient methods, we estimate the policy function directly. We need therefore a way to
represent it. We will use PyTorch to model the 



### Executing the policy

We saved the ```PolicyImpl``` model. We will load it on a Python script and run the environment in order to view how the agent is doing it.


## Summary

This example introduced the ```ReinforceSolver``` class that models the REINFORCE algorithm.

The RINFORCE algorithm, just like all policy gradient methods, suffers from high variance in the gradient estimation.
There are several reasons behind this high variance; e.g. sparse rewards or environment randomness.
Regardless of reasons behind high variance in the gradient, its effect can be detrimental during learning as it destabilizes it. Hence, reducing the high variance is important for feasible training. 
