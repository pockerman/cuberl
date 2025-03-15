#include "cubeai/base/cubeai_types.h"
#include "cubeai/maths/vector_math.h"
#include "cubeai/rl/policies/epsilon_greedy_policy.h"
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
using cuberl::rl::policies::EpsilonGreedyPolicy;
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
	
	EpsilonGreedyPolicy policy(eps);
	
	for(uint_t e=0; e < N_EPISODES; ++e){
	
		auto action = policy.get_action(max_rewards);
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


