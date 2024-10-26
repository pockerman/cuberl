#ifndef SIMPLE_REINFORCE_H
#define SIMPLE_REINFORCE_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/rl_algorithm_base.h"
#include "cubeai/rl/episode_info.h"
#include "cubeai/maths/vector_math.h"
#include "cubeai/data_structs/experience_buffer.h"

#include <torch/torch.h>

#include <vector>
#include <deque>
#include <numeric>
#include <iostream>
#include <chrono>
#include <memory>
#include <tuple>

namespace cubeai {
namespace rl {
namespace algos {
namespace pg {


///
/// \brief The ReinforceOpts struct. Holds various
/// configuration options for the Reinforce algorithm
///
struct ReinforceConfig
{
    uint_t max_num_of_episodes;
    uint_t max_itrs_per_episode;
    uint_t print_frequency;
    uint_t scores_queue_max_size;
    real_t gamma;
    real_t tolerance;
    real_t exit_score_level;

    ///
    /// \brief print
    /// \param out
    /// \return
    ///
    std::ostream& print(std::ostream& out)const;
};

template<typename ActionType, typename StateType>
struct ReinforceMonitor
{
	
	typedef StateType state_type;
	typedef ActionType action_type;
	typedef std::tuple<state_type, action_type, 
	                   real_t, state_type, bool,
					   real_t> experience_tuple_type;

	typedef cubeai::containers::ExperienceBuffer<experience_tuple_type> experience_buffer_type;
	
	/// monitor 
    std::vector<real_t> scores;
    std::deque<real_t> scores_deque;
    std::vector<real_t> saved_log_probs;
    std::vector<real_t> rewards;
	std::vector<real_t> policy_loss_values;
	std::vector<real_t> discounts;
	
	experience_buffer_type experience_buffer;
	
	
	void reset()noexcept;
	
	template<typename T, uint_t index>
    std::vector<T> 
    get(const std::vector<experience_tuple_type>& experience)const;
};


template<typename ActionType, typename StateType>
template<typename T, uint_t index>
std::vector<T> 
ReinforceMonitor<ActionType, StateType>::get(const std::vector<experience_tuple_type>& experience)const{
	
	std::vector<T> result;
	result.reserve(experience.size());
	
	auto b = experience.begin();
	auto e = experience.end();
	
	for(; b != e; ++b){
		auto item = *b;
		result.push_back(std::get<index>(item));
	}
	
	return result;
	
}


inline
std::ostream& operator<<(std::ostream& out, ReinforceConfig opts){
    return opts.print(out);
}

/**
  * @brief The ReinforceSolver class. The ReinforceSolver
  * trains a policy represented by the PolicyTp template parameter
  * on the environment represented by the EnvType parameter
  *
  */
template<typename EnvType, typename PolicyType, typename LossFuncType>
class ReinforceSolver final: public RLSolverBase<EnvType>
{
public:

    typedef EnvType env_type;
    typedef PolicyType policy_type;
	typedef LossFuncType loss_type;
	
	typedef typename env_type::state_type state_type;
	typedef typename env_type::action_type action_type;
	
    ///
    /// \brief Reinforce
    ///
    ReinforceSolver(ReinforceConfig opts, policy_type& policy,
	                loss_type& loss_fn,
                    std::unique_ptr<torch::optim::Optimizer>& policy_optimizer);

    ///
    /// \brief actions_before_training_begins. Execute any actions the
    /// algorithm needs before starting the iterations
    ///
    virtual void actions_before_training_begins(env_type&);

    ///
    /// \brief actions_after_training_ends. Actions to execute after
    /// the training iterations have finisehd
    ///
    virtual void actions_after_training_ends(env_type&){}

    ///
    /// \brief actions_before_training_episode
    ///
    virtual void actions_before_episode_begins(env_type&, uint_t /*episode_idx*/){}

    ///
    /// \brief actions_after_training_episode
    ///
    virtual void actions_after_episode_ends(env_type&, uint_t /*episode_idx*/,
                                            const EpisodeInfo& /*einfo*/);

    ///
    /// \brief on_episode Do one on_episode of the algorithm
    ///
    virtual EpisodeInfo on_training_episode(env_type&, uint_t /*episode_idx*/);

private:


    ///
    /// \brief config_
    ///
    ReinforceConfig config_;

    ///
    /// \brief policy_ptr_
    ///
    policy_type policy_ptr_;
	
	///
	/// \brief The loss function
	///
	loss_type loss_fn_;
	
    /**
     * @brief The policy_ optimzer
     */
    std::unique_ptr<torch::optim::Optimizer> policy_optimizer_;

	///
	/// \brief Helper class to monitor the algorithm
	///
	ReinforceMonitor<action_type, state_type> monitor_;

    

    ///
    /// \brief compute_discounts
    ///
    void compute_discounts_(std::vector<real_t>& data)const noexcept;

    ///
    /// \brief do_step_
    ///
    uint_t do_step_(env_type& env);

};

template<typename EnvType, typename PolicyType, typename LossFuncType>
ReinforceSolver<EnvType, PolicyType, LossFuncType>::ReinforceSolver(ReinforceConfig config,
                                                      policy_type& policy, 
                                                      loss_type& loss_fn, 
													  std::unique_ptr<torch::optim::Optimizer>& policy_optimizer)
    :
     RLSolverBase<EnvType>(),
     config_(config),
     policy_ptr_(policy),
	 loss_fn_(loss_fn),
     policy_optimizer_(std::move(policy_optimizer)),
	 monitor_()

{}

template<typename EnvType, typename PolicyType, typename LossFuncType>
void
ReinforceSolver<EnvType, PolicyType, LossFuncType>::actions_before_training_begins(env_type& /*env*/){

    scores_.clear();
    monitor_.reset();
	
	// the policy to train mode
    policy_ptr_ -> train();

}

template<typename EnvType, typename PolicyType, typename LossFuncType>
uint_t
ReinforceSolver<EnvType, PolicyType, LossFuncType>::do_step_(env_type& env){

	
	typedef ReinforceMonitor::experience_tuple_type experience_tuple_type;
	
    //  for every episode reset the environment
    auto old_timestep = env.reset().observation();

    uint_t itr = 0;
    for(; itr < config_.max_itrs_per_episode; ++itr){

      // from the policy get the action to do based
      // on the seen state
      auto [action, log_prob] = policy_ptr_ -> act(old_timestep.observation());
	  
      //monitor_.saved_log_probs.push_back(log_prob);
	  
      // execute the selected action on the environment
      auto new_timestep = env.step(action);
	  auto reward = new_timestep.reward()
	  
      // update state
      //auto new_state = time_step.observation();
	  
	  // keep track of the rewards
      //rewards_.push_back(time_step.reward());
	  
	  experience_tuple_type exp = {old_timestep.observation(), action, reward, 
	                               new_timestep.observation(), new_timestep.done(), 
								   log_prob};
	  
	  // put the observation into the buffer
	  monitor_.experience_buffer.append(exp);
	  
      if (new_timestep.done()){
          break;
      }
	  
	  old_timestep = new_timestep;
    }

    return itr;

}

template<typename EnvType, typename PolicyType, typename LossFuncType>
EpisodeInfo
ReinforceSolver<EnvType, PolicyType, LossFuncType>::on_training_episode(env_type& env, 
                                                                        uint_t episode_idx){
		
    // start the time for the episode																	
    auto start = std::chrono::steady_clock::now();

	// reset all the internal structures
	monitor_.reset();

	// Accummulate the data
    auto itrs = do_step_(env);
	
	typedef ReinforceMonitor::experience_tuple_type experience_tuple_type;
	typedef std::vector<experience_tuple_type> batch_type;
	
	auto batch = monitor_.experience_buffer.get<batch_type>();
	
	// create the batches
	// stack the experiences
	auto state_1_batch = monitor_.get<state_type,  0>(batch);
	auto action_batch  = monitor_.get<action_type, 1>(batch);
	auto reward_batch  = monitor_.get<real_t,      2>(batch);
	auto state_2_batch = monitor_.get<state_type,  3>(batch);
	auto done_batch    = monitor._get<bool, 4>(batch);
	auto log_probs_batch  = monitor_.get<real_t, 5>(batch);
					
	
    //auto rewards_sum = maths::sum(monitor_.rewards.begin(),
	//                              monitor_.rewards.end(),true);
	                              	                              

    //monitor_.scores_deque.push_back(rewards_sum);
    //monitor_.scores.push_back(rewards_sum);
    //monitor_.discounts.reserve(rewards_.size() + 1);

    //discounts = [self.gamma ** i for i in range(len(self.rewards) + 1)]
    //cubeai::maths::exponentiate(monitor_.discounts);

    // R = sum([a * b for a, b in zip(discounts, self.rewards)])
    /auto R = maths::dot_product(discounts.begin(), discounts.end(),
	//                            rewards_.begin(), rewards_.end()); //compute_total_reward_(discounts);

    //std::vector<real_t> policy_loss_values;
    //monitor_.policy_loss_values.reserve(saved_log_probs_.size());

    //for(auto& log_prob: saved_log_probs_){
    //    policy_loss_values.push_back(-log_prob * R);
    //}

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<real_t> elapsed_seconds = end - start;

    // the info class to return for the episode
    EpisodeInfo info;
    info.episode_index = episode_idx;
    info.episode_reward = R;
    info.episode_iterations = itrs;
    info.total_time = elapsed_seconds;
    return info;
}

template<typename EnvType, typename PolicyType, typename LossFuncType>
void
ReinforceSolver<EnvType, PolicyType, LossFuncType>::actions_after_episode_ends(env_type&, uint_t /*episode_idx*/,
                                                                               const EpisodeInfo& /*einfo*/){

     policy_optimizer_ -> zero_grad();
	 auto loss = loss_fn_(monitor_.policy_loss_values, monitor_.rewards);
     loss.backward();
     policy_optimizer_ -> step();


}

}
}
}
}
#endif
#endif // VANILLA_REINFORCE_H
