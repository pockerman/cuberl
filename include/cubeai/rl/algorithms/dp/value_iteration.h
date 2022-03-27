#ifndef VALUE_ITERATION_H
#define VALUE_ITERATION_H

#include "cubeai/base/cubeai_config.h" //KERNEL_PRINT_DBG_MSGS
#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dp/dp_algo_base.h"
#include "cubeai/rl/algorithms/dp/policy_improvement.h"
#include "cubeai/rl/algorithms/utils.h"
#include "cubeai/rl/episode_info.h"
#include "cubeai/io/csv_file_writer.h"

#include <memory>
#include <cmath>
#include <string>

#ifdef KERNEL_PRINT_DBG_MSGS
#include <iostream>
#endif

namespace cubeai{
namespace rl {
namespace algos {
namespace dp {

struct ValueIterationConfig
{
    uint_t n_max_iterations;
    real_t gamma;
    real_t tolerance;
    std::string save_path{""};
};

///
/// \brief ValueIteration class
///
template<typename EnvType, typename PolicyType, typename PolicyAdaptorType>
class ValueIteration: public DPAlgoBase<EnvType>
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
    /// \brief ValueIteration
    ///
    ValueIteration(const ValueIterationConfig config,
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
    virtual void actions_after_episode_ends(env_type&, uint_t /*episode_idx*/)override{}

    ///
    /// \brief on_episode Do one on_episode of the algorithm
    ///
    virtual EpisodeInfo on_training_episode(env_type& env, uint_t episode_idx) override;

    ///
    /// \brief policy_ptr
    /// \return
    ///
    std::shared_ptr<cubeai::rl::policies::DiscretePolicyBase> policy_ptr(){return  policy_;}

    ///
    ///
    ///
    void save(const std::string& filename)const;

private:

    ///
    /// \brief config_
    ///
    ValueIterationConfig config_;

    ///
    /// \brief v_
    ///
    DynVec<real_t> v_;

    ///
    /// \brief policy_
    ///
    policy_type& policy_;

    ///
    /// \brief policy_imp_
    ///
    PolicyImprovement<EnvType, PolicyType, PolicyAdaptorType> policy_imp_;

};

template<typename EnvType, typename PolicyType, typename PolicyAdaptorType>
ValueIteration<EnvType, PolicyType, PolicyAdaptorType>::ValueIteration(const ValueIterationConfig config,
                                                                       policy_type& policy,
                                                                       policy_adaptor_type& policy_adaptor)
    :
   DPAlgoBase<EnvType>(),
   config_(config),
   policy_(policy),
   policy_imp_(config.gamma, DynVec<real_t>(),  policy, policy_adaptor)
{}


template<typename EnvType, typename PolicyType, typename PolicyAdaptorType>
void
ValueIteration<EnvType, PolicyType, PolicyAdaptorType>::actions_before_training_begins(env_type& env){

    policy_imp_.actions_before_training_begins(env);
}

template<typename EnvType, typename PolicyType, typename PolicyAdaptorType>
EpisodeInfo
ValueIteration<EnvType, PolicyType, PolicyAdaptorType>::on_training_episode(env_type& env, uint_t episode_idx){

    EpisodeInfo info;
    auto delta = 0.0;
    for(uint_t s=0; s< env.n_states(); ++s){

        auto v = v_[s];
        auto max_val = blaze::max(state_actions_from_v(env, v_, config_.gamma, s));

        v_[s] = max_val;
        delta = std::max(delta, std::fabs(v_[s] - v));
    }

    // update residual
    //this->iter_controller_().update_residual( delta );

    if(delta < config_.tolerance){
        info.stop_training = true;
    }

    info.episode_index = episode_idx;
    return info;
}

template<typename EnvType, typename PolicyType, typename PolicyAdaptorType>
void
ValueIteration<EnvType, PolicyType, PolicyAdaptorType>::actions_after_training_ends(env_type& env){

    policy_imp_.set_value_function(v_);
    policy_imp_.on_training_episode(env, 0);
    policy_.update( policy_imp_.policy()); //.make_copy();

    if(config_.save_path != ""){
        save(config_.save_path);
    }

}

template<typename EnvType, typename PolicyType, typename PolicyAdaptorType>
void
ValueIteration<EnvType, PolicyType, PolicyAdaptorType>::save(const std::string& filename)const{

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

#endif // VALUE_ITERATION_H
