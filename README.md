# cuberl

Cuberl is a C++ library containing implementations of various reinforcement learning, filtering and planning algorithms.
The library documentation (under development) can be found here <a href="https://pockerman-py-cubeai.readthedocs.io/en/latest/">CubeAI</a>
The Python flavor of the library can be found at <a href="https://github.com/pockerman/py_cube_ai">PyCubeAI</a>. 

The following is an indicative list of examples. More tutorials can be found at <a href="https://pockerman-py-cubeai.readthedocs.io/en/latest/">CubeAI</a>.
 

## Examples

### Introductory

- <a href="examples/intro/intro_example_1/intro_example_1.md">_CubeRl_ Basics</a>
- <a href="examples/intro/intro_example_2/intro_example_2.md">Using PyTorch C++ API Part 1</a>
- <a href="examples/intro/intro_example_3/intro_example_3.md">Using PyTorch C++ API Part 2</a>

### Reinforcement learning

- <a href="https://pockerman-py-cubeai.readthedocs.io/en/latest/ExamplesCpp/rl/rl_example_0.html">Example 0: DummyAgent on  ```MountainCar-v0```</a>
- <a href="examples/example_1/example_1.cpp">Example 1: Toy Markov chain</a>
- <a href="examples/example_2/example_2.cpp">Example 2: Multi-armed bandit with epsilon-greedy policy</a>
- <a href="examples/example_3/example_3.cpp">Example 3: Multi-armed bandit with soft-max policy</a>
- <a href="examples/example_4/example_4.cpp">Example 4: Advertisement placement</a>
- <a href="examples/rl/rl_example_6/rl_example_6.cpp">Example 6: Iterative policy evaluation on ```FrozenLake-v0```</a>
- <a href="examples/rl/rl_example_7/rl_example_7.cpp">Example 7: Policy iteration on ```FrozenLake-v0```</a>
- <a href="examples/rl/rl_example_8/rl_example_8.cpp">Example 8: Value iteration on ```FrozenLake-v0```</a>
- <a href="examples/rl/rl_example_9/rl_example_9.cpp">Example 9: SARSA on ```CliffWalking-v0```</a>
- <a href="examples/rl/rl_example_10/rl_example_10.cpp">Example 10: Q-learning on ```CliffWalking-v0```</a>
- <a href="examples/rl/rl_example_11/rl_example_11.md">A2C on ```CartPole-v1```</a>
- <a href="examples/rl/rl_example_12/example_12.cpp">Example 12: DQN on ```CartPole-v0```</a>
- <a href="examples/rl/rl_example_13/rl_example_13.cpp">Example 13: Reinforce algorithm on ```CartPole-v0```</a>
- <a href="examples/rl/rl_example_14/rl_example_14.cpp">Example 14: Expected SARSA on ```CliffWalking-v0```</a>
- <a href="examples/example_15/example_15.cpp">Example 15: Approximate Monte Carlo on ```MountainCar-v0```</a>
- <a href="examples/example_16/example_16.cpp">Example 16: Monte Carlo tree search on ```Taxi-v3```</a>
- <a href="examples/rl/rl_example_18.cpp">Example 18: Double Q-learning on  ```CartPole-v0``` </a>
- <a href="examples/rl/rl_example_19/rl_example_19.cpp">Example 19: First visit Monte Carlo on ```FrozenLake-v0```</a>


### Path planning

- <a href="#">Example 19: Path planning with rapidly-exploring random trees (TODO)</a> 
- <a href="#">Example 20: Path planning with dynamic windows (TODO) </a>
- <a href="examples/example_17/example_17.cpp">Example 17: A* search on a road network  from Open Street Map data</a>

### PyTorch based examples

- <a href="examples/example_11/example_11.cpp">Example 11: Simple linear regression</a>


## Installation

The cubeai library has a host of dependencies:
Installation instructions and dependencies can be found <a href="https://pockerman-py-cubeai.readthedocs.io/en/latest/install.html">here</a>.

### Enabling PyTorch and CUDA

_cuberl_ can be complied with CUDA and/or PyTorch support. If PyTorch has been compiled using CUDA support, then
you need to enable CUDA as well. In order to do so set the flag _USE_CUDA_ in the _CMakeLists.txt_ to _ON_.
_cuberl_ assumes that PyTorch is compiled with the C++11 ABI.


### Documentation dependencies

There are extra dependencies if you want to generate the documentation. Namely,

- <a href="https://www.doxygen.nl/">Doxygen</a>
- <a href="https://www.sphinx-doc.org/en/master/">Sphinx</a>
- <a href="https://github.com/readthedocs/sphinx_rtd_theme">sphinx_rtd_theme</a>
- <a href="https://github.com/breathe-doc/breathe">breathe</a>
- <a href="https://github.com/crossnox/m2r2">m2r2</a>

Note that if Doxygen is not found on your system CMake will skip this. On a Ubuntu/Debian based machine, you can install
Doxygen using

```bash
sudo apt-get install doxygen
```

Similarly, install ```sphinx_rtd_theme``` using

```bash
pip install sphinx_rtd_theme
```

Install ```breathe``` using

```bash
pip install breathe
```

Install ```m2r2``` using

```bash
pip install m2r2
```


## Issues

#### undefined reference to ```cudaLaunchKernelExC@libcudart.so.11.0```. 

You may want to check with ```nvidia-msi``` your CUDA Version and make sure it is compatible with the PyTorch library you are linking against



