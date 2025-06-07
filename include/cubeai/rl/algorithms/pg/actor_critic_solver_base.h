#ifndef ACTOR_CRITIC_SOLVER_BASE_H
#define ACTOR_CRITIC_SOLVER_BASE_H

/// 
/// Implements synchronous  advantage-actor critic, A2C, algorithm
/// Currently the implementation of this class assumes that
/// PyTorch is used to model the deep networks
/// 
/// 

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
#include "cubeai/utils/torch_adaptor.h"
#include "cubeai/rl/algorithms/rl_algorithm_base.h"
#include "cubeai/rl/algorithms/utils.h"
#include "cubeai/rl/episode_info.h"
#include "cubeai/rl/algorithms/pg/a2c_config.h"
#include "cubeai/rl/algorithms/pg/a2c_monitor.h"
#include "cubeai/data_structs/experience_buffer.h"


#include <torch/torch.h>

#ifdef CUBERL_DEBUG
#include <cassert>
#endif

#include <string>
#include <chrono>
#include <map>
#include <any>
#include <memory>
#include <tuple>
#include <string>
#include <exception>
#include <iostream>

namespace cuberl{
namespace rl{
namespace algos {
namespace pg {

///
/// \brief A2C solver assuming separate networks for the actor and
/// the critic. 
/// Similar to the Reinforce algorithm implementation,
/// the PolicyType should expose an act method with the following signature
///  
///  std::tuple[ActioType, torch::Tensor> act(StateType)
///
///  In addition the CriticType should expose an evaluate method
///  with the following signature
///
///  torch::Tensor evaluate(StateType)
///
template<typename EnvType, typename PolicyType, 
         typename CriticType, typename MonitorType, 
		 typename ConfigType>
class ACSolverBase:  public RLSolverBase<EnvType>
{
public:

    ///
	/// \brief The environment type
	///
    typedef EnvType env_type;

    ///
	/// \brief The  policy or action type
	///
    typedef PolicyType policy_type;

    ///
	/// \brief The critic type
	///
    typedef CriticType critic_type;
	
	typedef typename env_type::state_type state_type;
	typedef typename env_type::action_type action_type;
	
	///
	/// \brief The monitor type to use
	///
	typedef MonitorType monitor_type;
	
	///
	/// \brief The solver configuration type
	///
	typedef ConfigType config_type;
	
	///
	/// \brief Destructor
	///
	virtual ~ACSolverBase()=default;
	
	///
    /// \brief actions_after_training_ends. Actions to execute after
    /// the training iterations have finisehd
    ///
    virtual void actions_after_training_ends(env_type&){}

    ///
    /// \brief actions_before_training_episode
    ///
    virtual void actions_before_episode_begins(env_type&, 
	                                           uint_t /*episode_idx*/){}

    ///
    /// \brief actions_after_training_episode
    ///
    virtual void actions_after_episode_ends(env_type&, 
	                                        uint_t /*episode_idx*/, 
	                                        const EpisodeInfo&){}
											
	///
    /// \brief actions_before_training_begins. Execute any actions the
    /// algorithm needs before starting the iterations
    ///
    virtual void actions_before_training_begins(env_type&);

	///
    /// \brief set_train_mode for both the Actor and the Critic
    ///
    void set_train_mode()noexcept;

    ///
    /// \brief set_evaluation_mode for both the Actor and the Critic
    ///
    void set_evaluation_mode()noexcept;
	
	///
	/// \brief Read-write access to the monitor object
	///
	monitor_type& get_monitor(){return monitor_;}
	
	
protected:

    ///
    /// \brief A2C
    /// \param config
    /// \param policy
    ///
    ACSolverBase(const config_type& config,
                 policy_type& policy, critic_type& critic,
                 std::unique_ptr<torch::optim::Optimizer>& policy_optimizer,
                 std::unique_ptr<torch::optim::Optimizer>& critic_optimizer);

    ///
    /// \brief config_
    ///
    config_type config_;

    ///
    /// \brief policy_
    ///
    policy_type& policy_;

    ///
	/// \brief The action network
	///
    critic_type& critic_;
	
	///
	/// \brief Helper class to monitor the algorithm
	///
	monitor_type monitor_;

    /// 
	/// \brief The policy_ optimzer
	/// 
    std::unique_ptr<torch::optim::Optimizer> policy_optimizer_;

    ///
	/// \brief The optimizer for the critic network
	///
    std::unique_ptr<torch::optim::Optimizer> critic_optimizer_;

};

template<typename EnvType, typename PolicyType, 
         typename CriticType, typename MonitorType, 
		 typename ConfigType>
ACSolverBase<EnvType, PolicyType, CriticType, 
             MonitorType, ConfigType>::ACSolverBase(const config_type& config,
                                                      policy_type& policy, critic_type& critic,
                                                      std::unique_ptr<torch::optim::Optimizer>& policy_optimizer,
                                                      std::unique_ptr<torch::optim::Optimizer>& critic_optimizer)
    :
	RLSolverBase<EnvType>(),
	config_(config),
	policy_(policy),
	critic_(critic),
	monitor_(),
	policy_optimizer_(std::move(policy_optimizer)),
	critic_optimizer_(std::move(critic_optimizer))
{}

template<typename EnvType, typename PolicyType, 
         typename CriticType, typename MonitorType, 
		 typename ConfigType>
void
ACSolverBase<EnvType, PolicyType, CriticType, 
             MonitorType, ConfigType>::set_train_mode()noexcept{
    policy_ -> train();
    critic_ -> train();

}

template<typename EnvType, typename PolicyType, 
         typename CriticType, typename MonitorType, 
		 typename ConfigType>
void
ACSolverBase<EnvType, PolicyType, CriticType, 
             MonitorType, ConfigType>::set_evaluation_mode()noexcept{
    policy_ -> eval();
    critic_ -> eval();

}

template<typename EnvType, typename PolicyType, 
         typename CriticType, typename MonitorType, 
		 typename ConfigType>
void
ACSolverBase<EnvType, PolicyType, CriticType, 
             MonitorType, ConfigType>::actions_before_training_begins(env_type& /*env*/){
	
	monitor_.reset();
	monitor_.policy_loss_values.reserve(config_.n_episodes);
	monitor_.critic_loss_values.reserve(config_.n_episodes);
	monitor_.rewards.reserve(config_.n_episodes);
	monitor_.episode_duration.reserve(config_.n_episodes);
    set_train_mode();
}

}
}
}
}

#endif // USE_PYTORCH
#endif 
