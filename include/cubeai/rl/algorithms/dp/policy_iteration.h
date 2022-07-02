#ifndef POLICY_ITERATION_H
#define POLICY_ITERATION_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dp/dp_algo_base.h"
#include "cubeai/rl/algorithms/dp/iterative_policy_evaluation.h"
#include "cubeai/rl/algorithms/dp/policy_improvement.h"

#include <string>

namespace cubeai{
namespace rl{
namespace algos {
namespace dp{


///
/// \brief The PolicyIterationConfig struct
///
struct PolicyIterationConfig
{
    uint_t n_policy_eval_steps;
    real_t gamma{1.0};
    real_t tolerance{1.0e-6};
    std::string save_path{""};
};

///
/// \brief The policy iteration class
///
template<typename EnvType, typename PolicyType, typename PolicyAdaptorType>
class PolicyIteration final: public DPAlgoBase<EnvType>
{
public:

    ///
    /// \brief env_t
    ///
    typedef typename DPAlgoBase<EnvType>::env_type env_type;

    ///
    /// \brief policy_type
    ///
    typedef PolicyType policy_type;

    ///
    /// \brief policy_adaptor_type
    ///
    typedef PolicyAdaptorType policy_adaptor_type;

    ///
    /// \brief PolicyIteration
    ///
    PolicyIteration(PolicyIterationConfig config,
                    policy_type& policy,
                    policy_adaptor_type& policy_adaptor);

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
    virtual void actions_after_episode_ends(env_type&, uint_t /*episode_idx*/, const EpisodeInfo& /*einfo*/)override{}

    ///
    /// \brief on_episode Do one on_episode of the algorithm
    ///
    virtual EpisodeInfo on_training_episode(env_type& env, uint_t episode_idx) override;

    ///
    /// \brief save
    /// \param filename
    ///
    void save(const std::string& filename)const;

private:

    PolicyIterationConfig config_;

    ///
    /// \brief v_
    ///
    DynVec<real_t> v_;

    ///
    /// \brief policy_eval_
    ///
    IterativePolicyEval<env_type, policy_type> policy_eval_;


    ///
    /// \brief policy_imp_
    ///
    PolicyImprovement<env_type, policy_type, policy_adaptor_type> policy_imp_;

};

template<typename EnvType, typename PolicyType, typename PolicyAdaptorType>
PolicyIteration<EnvType, PolicyType, PolicyAdaptorType>::PolicyIteration(PolicyIterationConfig config,
                                                                         policy_type& policy,
                                                                         policy_adaptor_type& policy_adaptor)
    :
    DPAlgoBase<EnvType>(),
    config_(config),
    v_(),
    policy_eval_({config.gamma, config.tolerance}, policy),
    policy_imp_(config.gamma, DynVec<real_t>(), policy, policy_adaptor)
{}


template<typename EnvType, typename PolicyType, typename PolicyAdaptorType>
void
PolicyIteration<EnvType, PolicyType, PolicyAdaptorType>::actions_before_training_begins(env_type& env){

    policy_eval_.actions_before_training_begins(env);
    policy_imp_.actions_before_training_begins(env);
}

template<typename EnvType, typename PolicyType, typename PolicyAdaptorType>
void
PolicyIteration<EnvType, PolicyType, PolicyAdaptorType>::actions_after_training_ends(env_type& /*env*/){
    v_ = policy_eval_.get_value_function();

    if(config_.save_path != ""){
        save(config_.save_path);
    }
}

template<typename EnvType, typename PolicyType, typename PolicyAdaptorType>
EpisodeInfo
PolicyIteration<EnvType, PolicyType, PolicyAdaptorType>::on_training_episode(env_type& env, uint_t episode_idx){

    auto start = std::chrono::steady_clock::now();
    EpisodeInfo info;

    auto episode_rewards = 0.0;

    // make a copy of the policy already obtained
    auto old_policy = policy_eval_.get_policy();

    for(uint_t itr=0; itr < config_.n_policy_eval_steps; ++itr ){
        // evaluate the policy
        policy_eval_.on_training_episode(env, itr);
    }

    // update the value function to
    // improve for
    policy_imp_.set_value_function( policy_eval_.get_value_function());

    // improve the policy
    policy_imp_.on_training_episode(env, episode_idx);

    const auto& new_policy = policy_imp_.policy();

    // policy converged
    if(old_policy == new_policy){
        info.stop_training = true;
    }

    policy_eval_.update_policy(new_policy);
    //policy_eval_.reset();
    //policy_imp_.reset();

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<real_t> elapsed_seconds = end-start;


    info.episode_index = episode_idx;
    info.episode_reward = episode_rewards;
    //info.episode_iterations = itr_counter.current_iteration_index();
    info.total_time = elapsed_seconds;

    return info;
}

template<typename EnvType, typename PolicyType, typename PolicyAdaptorType>
void
PolicyIteration<EnvType, PolicyType, PolicyAdaptorType>::save(const std::string& filename)const{

    CSVWriter file_writer(filename, ',', true);
    file_writer.write_column_names({"state_index", "value_function"});

    for(uint_t s=0; s < v_.size(); ++s){
        auto row = std::make_tuple(s, v_[s]);
        file_writer.write_row(row);
    }
}

}
}
}
}

#endif // POLICY_ITERATION_H
