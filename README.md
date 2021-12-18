# cubeAI

CubeAI is a C++ library containing implementations of various reinforcement learning, filtering and planning algorithms
 

## Examples

- <a href="examples/example_1/example_1.cpp">Example 1: Toy Markov chain</a>
- <a href="examples/example_2/example_2.cpp">Example 2: Multi-armed bandit with epsilon-greedy policy</a>
- <a href="examples/example_3/example_3.cpp">Example 3: Multi-armed bandit with soft-max policy</a>
- <a href="examples/example_6/example_6.cpp">Example 6: Iterative policy evaluation on ```FrozenLake-v0```</a>
- <a href="examples/example_7/example_7.cpp">Example 7: Policy iteration on ```FrozenLake-v0```</a>
- <a href="examples/example_8/example_8.cpp">Example 8: Value iteration on ```FrozenLake-v0```</a>
- <a href="examples/example_9/example_9.cpp">Example 9: SARSA on ```CliffWalking-v0```</a>
- <a href="examples/example_10/example_10.cpp">Example 10: Q-learning on ```CliffWalking-v0```</a>
- <a href="examples/example_14/example_14.cpp">Example 14: Expected SARSA on ```CliffWalking-v0```</a>
- <a href="#">Example 9: </a> A* search
- <a href="#">TODO: </a> D* search
- <a href="#">Example 29:</a> Path planning with rapidly-exploring random trees
- <a href="#">Example 35:</a> Path planning with dynamic windows  


### PyTorch based examples

- <a href="examples/example_11/example_11.cpp">Example 11: Simple linear regression</a>
- <a href="examples/example_12/example_12.cpp">Example 12: DQN on ```CartPole-v0```</a>
- <a href="examples/example_13/example_13.cpp">Example 13: Reinforce algorithm on ```CartPole-v0```</a>


## Dependencies

- CMake
- PyTorch
- Blaze
- gymfcpp

## Installation

- Install the dependencies
- ```mkdir build && cd build```
- ```cmake ..```
- ```make install```

If you are using ```gymfcpp``` you need to export the path to the Python version you are using. For ecample:

```
export CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/python3.8/"
```



## References

- <a href="https://pytorch.org/cppdocs/">PyTorch C++ API</a>


## Images

![Following a path](images/path_following.gif "Following a path")

![State value function](images/state_value_function.png "State value function")

