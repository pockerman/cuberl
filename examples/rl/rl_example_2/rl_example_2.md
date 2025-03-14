# Example 2: Multi-armed bandits

In this example we will look at how to solve the multi-armed bandits problem.
The multi-armed bandits framework has a lot of application to online decision making such as 
advertisement placement, and recommendation engines. The book 
<a href="https://www.oreilly.com/library/view/bandit-algorithms-for/9781449341565/">Bandit Algorithms for Website Optimization</a>
by John Myles White, discusses bandit algorithms for website optimization</a>.

In this example we will simulate the multi-armed bandits problem using the following 
sampling methodologies $\epsilon$-greedy and Thompson sampling.

----
**Remark**

The article, by  James LeDoux, <a href="https://jamesrledoux.com/algorithms/bandit-algorithms-epsilon-ucb-exp-python/">
Multi-Armed Bandits in Python: Epsilon Greedy, UCB1, Bayesian UCB, and EXP3</a>

provides a nice overview of $\epsilon$-greedy and Thompson sampling

----

### $\epsilon$-greedy

### Thompson sampling


## Driver code

```
#include "cubeai/base/cubeai_types.h"
#include "cubeai/maths/vector_math.h"

#include "rlenvs/utils/maths/statistics/distributions/beta_dist.h"
#include "rlenvs/utils/maths/statistics/distributions/bernoulli_dist.h"
#include "rlenvs/envs/multi_armed_bandits/multi_armed_bandits.h"

#include <cmath>
#include <utility>
#include <tuple>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>
#include <utility>
#include <unordered_map>

namespace rl_example_2
{

using cuberl::real_t;
using cuberl::uint_t;
using cuberl::int_t;
using cuberl::DynMat;
using cuberl::DynVec;

using rlenvscpp::utils::maths::stats::BetaDist;
using rlenvscpp::utils::maths::stats::BernoulliDist;
using rlenvscpp::envs::bandits::MultiArmedBandits;

// Number of episodes to play
uint_t N_EPISODES = 1000;
real_t EPS = 0.2;

struct SamplingResult
{
	real_t total_reward;
	DynVec<uint_t> lever_pulls;
	
};

SamplingResult 
run_epsilon_greedy(MultiArmedBandits& env, real_t eps){
	
	SamplingResult result;
	result.total_reward = 0.0;
	result.lever_pulls.resize(env.n_actions());
	
	DynVec<real_t> max_rewards(env.n_actions());
	for(uint_t a=0; a < env.n_actions(); ++a){
		max_rewards[a] = 0.0;
		result.lever_pulls[a] = 0;
	}
	
	// generate a number in [0, 1]
	std::uniform_real_distribution<> real_dist_(0.0, 1.0);

	// uniformly select an action
	std::uniform_int_distribution<int_t> distribution(0, 
	                                               static_cast<int>(env.n_actions())-1);
												   
	std::random_device rd;
	std::mt19937 generator(rd());
	
	for(uint_t e=0; e < N_EPISODES; ++e){
		
		auto action = 0;
		if(real_dist_(generator) > eps){
			action = cuberl::maths::arg_max(max_rewards);
		}
		else{
			
			// randomly select an action
			action = distribution(generator);
		}
		
		
		auto time_step = env.step(action);
		
		// get the reward from the action
		auto reward = time_step.reward();
		
		// update the table
		max_rewards[action] += reward;
		
		result.total_reward += reward;
		result.lever_pulls[action] += 1;
	}
	
	return result;
	
}

SamplingResult 
run_thompson_sampling(MultiArmedBandits& env){
	
	SamplingResult result;
	result.total_reward = 0.0;
	result.lever_pulls.resize(env.n_actions());
	
	for(uint_t a=0; a < env.n_actions(); ++a){
		result.lever_pulls[a] = 0;
	}
	
	// every arm is modelled using beta dist
	std::vector<BetaDist<real_t>> beta_dists(env.n_actions(), 
	                                 BetaDist<real_t>(1.0, 1.0));
	
	std::vector<real_t> samples(env.n_actions(), 0.0);
	
	for(uint_t e=0; e < N_EPISODES; ++e){
		
		env.reset();
		
		// choose which lever to pull
		for(uint_t i=0; i < beta_dists.size(); ++i){
			samples[i] = beta_dists[i].sample();
		}
		
		// find the max
		auto max_item = cuberl::maths::arg_max(samples);
		auto time_step = env.step(max_item);
		
		auto reward = time_step.reward();
		result.total_reward += reward;
		
		auto alpha = beta_dists[max_item].alpha();
		auto beta =  beta_dists[max_item].beta();
		
		// update the alpha param
		if(reward == env.success_reward()){
			alpha += 1.0;
		}
		else{
			
			beta += 1.0;
		}
		
		// reset the distribution
		beta_dists[max_item].reset(alpha, beta);
		result.lever_pulls[max_item] += 1;
			
	}
	
	return result;
}

}

int main() {

    using namespace rl_example_2;
	
	{
	
		// the environment
		MultiArmedBandits env;
		
		/// all levers have the same probability of success
		// so there shouldn't be any preferance
		std::vector<real_t> p(5, 0.5);
		std::unordered_map<std::string, std::any> options;
		options["p"] = std::any(p);
		env.make("v0", options);
		env.reset();
		
		std::cout<<"Running thompson sampling"<<std::endl;
		auto thompson_result = run_thompson_sampling(env);
		
		std::cout<<"Thompson reward: "<<thompson_result.total_reward<<std::endl;
		std::cout<<"Thompson selected levers: "<<thompson_result.lever_pulls<<std::endl;
		
		std::cout<<"Running epsilon greedy sampling"<<std::endl;
		auto epsilon_greedy_result = run_epsilon_greedy(env,EPS);
		
		std::cout<<"epsilon-greedy reward: "<<epsilon_greedy_result.total_reward<<std::endl;
		std::cout<<"epsilon-greedy selected levers: "<<epsilon_greedy_result.lever_pulls<<std::endl;
	}
	
	std::cout<<"========================================"<<std::endl;
	{
	
		// the environment
		MultiArmedBandits env;
		
		// levers have different probability
		// of success 
		std::vector<real_t> p(5, 0.2);
		p[0] = 0.1;
		p[1] = 0.2;
		p[2] = 0.3;
		p[3] = 0.4;
		p[4] = 0.5;
		std::unordered_map<std::string, std::any> options;
		options["p"] = std::any(p);
		env.make("v0", options);
		env.reset();
		
		std::cout<<"Running thompson sampling"<<std::endl;
		auto thompson_result = run_thompson_sampling(env);
		
		std::cout<<"Thompson reward: "<<thompson_result.total_reward<<std::endl;
		std::cout<<"Thompson selected levers: "<<thompson_result.lever_pulls<<std::endl;
		
		std::cout<<"Running epsilon greedy sampling"<<std::endl;
		auto epsilon_greedy_result = run_epsilon_greedy(env,EPS);
		
		std::cout<<"epsilon-greedy reward: "<<epsilon_greedy_result.total_reward<<std::endl;
		std::cout<<"epsilon-greedy selected levers: "<<epsilon_greedy_result.lever_pulls<<std::endl;
	}

   return 0;
}

```

Running the driver above produces the following output (this may be different on your machine.

```
Running thompson sampling
Thompson reward: 492
Thompson selected levers: 358  78  13 299 252
Running epsilon greedy sampling
epsilon-greedy reward: 526
epsilon-greedy selected levers: 852  32  36  34  46
========================================
Running thompson sampling
Thompson reward: 468
Thompson selected levers:  27  42  61  90 780
Running epsilon greedy sampling
epsilon-greedy reward: 141
epsilon-greedy selected levers: 814  36  48  54  48

```


We can see that when all levers have the same probability of success Thompson sampling and
$\epsilon$-greedy perform more or less the same, with $\epsilon$-greedy having some better performance.
However, when the levers have different probability of success, Thompson sampling outperforms
$\epsilon$-greedy by roughly a factor of 4. Notice that we can choose to force the $\epsilon$
to decay as we do more steps assuming which means will get more exploitation. Most likely,
this will have better performance. 


## Summary

This example implemented Thompson sampling and 
