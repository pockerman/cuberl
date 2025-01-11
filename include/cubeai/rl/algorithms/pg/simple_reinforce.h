#ifndef SIMPLE_REINFORCE_H
#define SIMPLE_REINFORCE_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/rl_algorithm_base.h"
#include "cubeai/rl/episode_info.h"
#include "cubeai/maths/vector_math.h"
#include "cubeai/data_structs/experience_buffer.h"
# include "cubeai/utils/torch_adaptor.h"

#include <torch/torch.h>

#include <vector>
#include <deque>
#include <numeric>
#include <iostream>
#include <chrono>
#include <memory>
#include <tuple>
#include <string>

namespace cuberl {
namespace rl {
namespace algos {
namespace pg {


///
/// \brief The ReinforceOpts struct. Holds various
/// configuration options for the Reinforce algorithm
///
struct ReinforceConfig
{
	bool normalize_rewards{false};
    uint_t max_num_of_episodes;
    uint_t max_itrs_per_episode;
    uint_t print_frequency;
    uint_t scores_queue_max_size;
    real_t gamma;
    real_t tolerance;
    real_t exit_score_level;
	
	DeviceType device_type;

    ///
    /// \brief print
    /// \param out
    /// \return
    ///
    std::ostream& print(std::ostream& out)const;
	
	///
	/// \brief Load the configuration from the given json file
	///
	void load_from_json(const std::string& filename);
};

template<typename ActionType, typename StateType>
struct ReinforceMonitor
{
	
	typedef StateType state_type;
	typedef ActionType action_type;
	typedef std::tuple<state_type, action_type, 
	                   real_t, state_type, bool,
					   real_t> experience_tuple_type;

	typedef cuberl::containers::ExperienceBuffer<experience_tuple_type> experience_buffer_type;
	
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

template<typename ActionType, typename StateType>
void 
ReinforceMonitor<ActionType, StateType>::reset()noexcept{

    std::vector<real_t> empty;
    std::swap(saved_log_probs, empty);
    empty.clear();
    std::swap(rewards, empty);
	
	empty.clear();
	std::swap(policy_loss_values, empty);
	
	empty.clear();
	std::swap(discounts, empty);

    std::deque<real_t> empty_deque;
    std::swap(scores_deque, empty_deque);
	
	experience_buffer.clear();

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
    ReinforceSolver(ReinforceConfig opts, 
	                policy_type& policy,
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

    monitor_.reset();
	
	// set the policy to train mode
    policy_ptr_ -> train();

}

template<typename EnvType, typename PolicyType, typename LossFuncType>
uint_t
ReinforceSolver<EnvType, PolicyType, LossFuncType>::do_step_(env_type& env){

	
	typedef typename ReinforceMonitor<action_type, state_type>::experience_tuple_type experience_tuple_type;
	
    //  for every episode reset the environment
    auto old_timestep = env.reset();

    uint_t itr = 0;
    for(; itr < config_.max_itrs_per_episode; ++itr){

      // from the policy get the action to do based
      // on the seen state
      auto [action, log_prob] = policy_ptr_ -> act(old_timestep.observation());
	  
      // execute the selected action on the environment
      auto new_timestep = env.step(action);
	  auto reward = new_timestep.reward();
	  
	  experience_tuple_type exp = {old_timestep.observation(), 
	                               action, 
								   reward, 
	                               new_timestep.observation(), 
								   new_timestep.done(), 
								   log_prob};
	  
	  // put the observation into the buffer
	  monitor_.experience_buffer.append(exp);
	  
      if (new_timestep.done()){
          break;
      }
	  
	  old_timestep = new_timestep;
    }

	// because we start from zero
    return itr + 1;
}

template<typename EnvType, typename PolicyType, typename LossFuncType>
EpisodeInfo
ReinforceSolver<EnvType, PolicyType, LossFuncType>::on_training_episode(env_type& env, 
                                                                        uint_t episode_idx){
	
	
    // start the time for the episode																	
    auto start = std::chrono::steady_clock::now();

	// reset all the internal structures
	monitor_.reset();

	// Accummulate the data i.e. create the
	// batch data we need to train the parameters
    auto itrs = do_step_(env);
	
	typedef typename ReinforceMonitor<action_type, state_type>::experience_tuple_type experience_tuple_type;
	typedef std::vector<experience_tuple_type> batch_type;
	
	// the accumulated batch
	auto batch = monitor_.experience_buffer.template get<batch_type>();
	
	// create the batches
	// stack the experiences
	auto state_1_batch = monitor_.template get<state_type,  0>(batch);
	auto action_batch  = monitor_.template get<action_type, 1>(batch);
	auto reward_batch  = monitor_.template get<real_t,      2>(batch);
	auto state_2_batch = monitor_.template get<state_type,  3>(batch);
	auto done_batch    = monitor_.template get<bool, 4>(batch);
	auto log_probs_batch  = monitor_.template get<real_t, 5>(batch);
	
	auto tensor_log_probs_batch = cuberl::utils::pytorch::TorchAdaptor::to_torch(log_probs_batch, 
														                      config_.device_type,
																			  true);
	
	
	auto discounted_coeffs = cuberl::maths::exponentiate(reward_batch, config_.gamma);
	auto discounted_returns = cuberl::maths::element_product(reward_batch, discounted_coeffs);
	
	if(config_.normalize_rewards){
		
		discounted_returns = cuberl::maths::normalize_max(discounted_returns);
	}
	
	// TODO: These should be the discounted rewards
	auto tensor_reward_batch = cuberl::utils::pytorch::TorchAdaptor::to_torch(discounted_returns, 
													                       config_.device_type, true);
	
	// now that we have the batches
	policy_optimizer_ -> zero_grad();
	auto loss = loss_fn_(tensor_log_probs_batch, tensor_reward_batch);
	loss.backward();
	policy_optimizer_ -> step();
					
	
    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<real_t> elapsed_seconds = end - start;

    // the info class to return for the episode
    EpisodeInfo info;
    info.episode_index = episode_idx;
    //info.episode_reward = R;
    info.episode_iterations = itrs;
    info.total_time = elapsed_seconds;
    return info;

}

template<typename EnvType, typename PolicyType, typename LossFuncType>
void
ReinforceSolver<EnvType, PolicyType, LossFuncType>::actions_after_episode_ends(env_type&, uint_t /*episode_idx*/,
                                                                               const EpisodeInfo& /*einfo*/){}

}
}
}
}
#endif
#endif // VANILLA_REINFORCE_H
