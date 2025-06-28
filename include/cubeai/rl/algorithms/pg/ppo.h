#ifndef PPO_H
#define PPO_H

/// 
/// Implements Proximal Policy Optimization algorithm
/// Currently the implementation of this class assumes that
/// PyTorch is used to model the deep networks
/// 
/// 

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
#include "cubeai/utils/torch_adaptor.h"
#include "cubeai/rl/algorithms/rl_algorithm_base.h"
#include "cubeai/rl/algorithms/pg/actor_critic_solver_base.h"
#include "cubeai/rl/algorithms/utils.h"
#include "cubeai/rl/episode_info.h"
#include "cubeai/rl/algorithms/pg/ppo_config.h"
#include "cubeai/rl/algorithms/pg/a2c_monitor.h"
#include "cubeai/data_structs/experience_buffer.h"


#include <torch/torch.h>

#ifdef CUBERL_DEBUG
#include <cassert>
#endif

namespace cuberl{
namespace rl{
namespace algos {
namespace pg {
	
template<typename EnvType, typename PolicyType, typename CriticType>
class PPOSolver final: public ACSolverBase<EnvType, PolicyType, 
										   CriticType, 
										   A2CMonitor<typename EnvType::action_type,
										              typename EnvType::state_type>,
										   PPOConfig>
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
    /// \brief PPOSolver Constructor
    ///
    PPOSolver(const PPOConfig& config,
              policy_type& policy, critic_type& critic,
              std::unique_ptr<torch::optim::Optimizer>& policy_optimizer,
              std::unique_ptr<torch::optim::Optimizer>& critic_optimizer);
			  
	///
    /// \brief on_episode Do one on_episode of the algorithm
    ///
    virtual EpisodeInfo on_training_episode(env_type&, uint_t /*episode_idx*/) override final;
	
	
	
};
	
template<typename EnvType, typename PolicyType, typename CriticType>
PPOSolver<EnvType, PolicyType, CriticType>::PPOSolver(const PPOConfig& config,
													  policy_type& policy, critic_type& critic,
													  std::unique_ptr<torch::optim::Optimizer>& policy_optimizer,
													  std::unique_ptr<torch::optim::Optimizer>& critic_optimizer)
			  :
			  ACSolverBase<EnvType, PolicyType, CriticType, 
							A2CMonitor<typename EnvType::action_type, typename EnvType::state_type>,
							PPOConfig>(config, policy, critic, policy_optimizer, critic_optimizer)
{}

template<typename EnvType, typename PolicyType, typename CriticType>
EpisodeInfo 
PPOSolver<EnvType, PolicyType, CriticType>::on_training_episode(env_type&, uint_t /*episode_idx*/)
{
	
}


}// pg
}// algos
}// rl
}// cuberl
#endif
#endif