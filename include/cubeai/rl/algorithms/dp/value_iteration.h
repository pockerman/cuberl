#ifndef VALUE_ITERATION_H
#define VALUE_ITERATION_H

#include "cubeai/base/cubeai_config.h" //KERNEL_PRINT_DBG_MSGS
#include "cubeai/base/cubeai_types.h"

#include "cubeai/rl/algorithms/dp/dp_algo_base.h"
#include "cubeai/rl/algorithms/dp/policy_improvement.h"
#include "cubeai/rl/algorithms/utils.h"
#include "cubeai/rl/episode_info.h"
#include "cubeai/rl/policies/max_tabular_policy.h"
#include "rlenvs/utils/io/csv_file_writer.h"
#include "rlenvs/rlenvs_consts.h"

#include <memory>
#include <cmath>
#include <string>


namespace cuberl{
namespace rl {
namespace algos {
namespace dp {

///
/// \brief The ValueIterationConfig struct
///
struct ValueIterationConfig
{
    real_t gamma{1.0};
    real_t tolerance {rlenvscpp::consts::TOLERANCE};
    std::string save_path{rlenvscpp::consts::INVALID_STR};
};

///
/// \brief ValueIteration class
///
template<typename EnvType>
class ValueIteration: public DPSolverBase<EnvType>
{
public:

    ///
    /// \brief env_t
    ///
    typedef typename DPSolverBase<EnvType>::env_type env_type;

    ///
    /// \brief ValueIteration
    ///
    ValueIteration(const ValueIterationConfig config);

    ///
    /// \brief actions_before_training_begins. Execute any actions the
    /// algorithm needs before starting the iterations
    ///
    virtual void actions_before_training_begins(env_type& env)override;

    ///
    /// \brief actions_after_training_ends. Actions to execute after
    /// the training iterations have finisehd
    ///
    virtual void actions_after_training_ends(env_type& /*env*/)override;

    ///
    /// \brief actions_before_training_episode
    ///
    virtual void actions_before_episode_begins(env_type&, uint_t /*episode_idx*/)override{}

    ///
    /// \brief actions_after_training_episode
    ///
    virtual void actions_after_episode_ends(env_type&, uint_t /*episode_idx*/,
                                            const EpisodeInfo& /*einfo*/)override{}

    ///
    /// \brief on_episode Do one on_episode of the algorithm
    ///
    virtual EpisodeInfo on_training_episode(env_type& env, uint_t episode_idx) override;

    ///
    ///
    ///
    void save(const std::string& filename)const;
	
	
	///
	/// \brief
	///
	cuberl::rl::policies::MaxTabularPolicy build_policy(const env_type& env)const;
	
private:

    ///
    /// \brief config_
    ///
    ValueIterationConfig config_;

    ///
    /// \brief v_
    ///
    DynVec<real_t> v_;

};

template<typename EnvType>
ValueIteration<EnvType>::ValueIteration(const ValueIterationConfig config)
    :
   DPSolverBase<EnvType>(),
   config_(config)
{}


template<typename EnvType>
void
ValueIteration<EnvType>::actions_before_training_begins(env_type& env){
	v_ = DynVec<real_t>::Zero(env.n_states());	
}

template<typename EnvType>
EpisodeInfo
ValueIteration<EnvType>::on_training_episode(env_type& env, 
											 uint_t episode_idx){

	// start timing the training
    auto start = std::chrono::steady_clock::now();
	
    EpisodeInfo info;
    auto delta = 0.0;
    for(uint_t s=0; s< env.n_states(); ++s){

        auto v = v_[s];
        auto max_val = state_actions_from_v(env, v_, config_.gamma, s).maxCoeff();

        v_[s] = max_val;
        delta = std::max(delta, std::fabs(v_[s] - v));
    }

	// inform the outer loop that 
	// we converged
    if(delta < config_.tolerance){
        info.stop_training = true;
    }
	
	auto end = std::chrono::steady_clock::now();
    std::chrono::duration<real_t> elapsed_seconds = end-start;

    info.episode_index = episode_idx;
	info.episode_iterations = env.n_states();
	info.total_time = elapsed_seconds;
	
	// this is artificial but helps
	// to monitor convergence
	info.episode_reward = delta;
	
    return info;
}

template<typename EnvType>
void
ValueIteration<EnvType>::actions_after_training_ends(env_type&){
    if(config_.save_path != rlenvscpp::consts::INVALID_STR){
        save(config_.save_path);
    }

}

template<typename EnvType>
void
ValueIteration<EnvType>::save(const std::string& filename)const{

    rlenvscpp::utils::io::CSVWriter file_writer(filename, ',');
    file_writer.open();

    file_writer.write_column_names({"state_index", "value_function"});

    for(uint_t s=0; s < static_cast<uint_t>(v_.size()); ++s){
        auto row = std::make_tuple(s, v_[s]);
        file_writer.write_row(row);
    }
}

template<typename EnvType>
cuberl::rl::policies::MaxTabularPolicy 
ValueIteration<EnvType>::build_policy(const env_type& env)const{
	
	cuberl::rl::policies::MaxTabularPolicy policy;
	cuberl::rl::policies::MaxTabularPolicyBuilder builder;
	builder.build_from_state_function(env, v_,
									  config_.gamma,policy);
	return policy;
	
}

}
}
}
}

#endif // VALUE_ITERATION_H
