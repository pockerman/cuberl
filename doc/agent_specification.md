# Agent specification

An agent in cubeai should abide with the following constraints

- Implemented as a class
- The class is a template class which specifies the environment type
- The class implements the following methods
	- ```std::tuple<uint_t, real_t> on_episode()```
	- ```void actions_before_training_episodes()```
	- ```void actions_after_training_episodes()```
	- ```void actions_before_training_episode()```
	- ```void actions_after_training_episode()```
	- ```const DynVec<real_t>& get_rewards()const```
	- ```DynVec<real_t>& get_rewards()```
	- ```const std::vector<uint_t>& get_iterations()const```
	- ```std::vector<uint_t>& get_iterations()```
	
