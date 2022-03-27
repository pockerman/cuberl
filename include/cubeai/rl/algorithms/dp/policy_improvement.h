#ifndef POLICY_IMPROVEMENT_H
#define POLICY_IMPROVEMENT_H

#include "cubeai/rl/algorithms/dp/dp_algo_base.h"
#include "cubeai/rl/algorithms/utils.h"
#include "cubeai/rl/policies/policy_adaptor_base.h"
#include "cubeai/rl/policies/discrete_policy_base.h"

#include <memory>
#include <any>
#include <map>
#include <string>

namespace cubeai{
namespace rl {
namespace algos {
namespace dp {

///
/// \brief The PolicyImprovement class
///
template<typename EnvType, typename PolicyType, typename PolicyAdaptorType>
class PolicyImprovement: public DPAlgoBase<EnvType>
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
    /// \brief IterativePolicyEval
    ///
    PolicyImprovement(real_t gamma, const DynVec<real_t>& val_func,
                      policy_type& policy,
                      policy_adaptor_type& policy_adaptor);

    ///
    /// \brief actions_before_training_begins. Execute any actions the
    /// algorithm needs before starting the iterations
    ///
    virtual void actions_before_training_begins(env_type& /*env*/)override{}

    ///
    /// \brief actions_after_training_ends. Actions to execute after
    /// the training iterations have finisehd
    ///
    virtual void actions_after_training_ends(env_type& /*env*/)override{}

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
    /// \brief policy
    /// \return
    ///
    const policy_type& policy()const{return  policy_;}

    ///
    /// \brief policy
    /// \return
    ///
    policy_type& policy(){return  policy_;}

    ///
    /// \brief set_value_function
    /// \param v
    ///
    void set_value_function(const DynVec<real_t>& v){v_ = v;}

protected:

    ///
    /// \brief gamma_
    ///
    real_t gamma_;

    ///
    /// \brief v_
    ///
    DynVec<real_t> v_;

    ///
    /// \brief policy_
    ///
    policy_type& policy_;

    ///
    /// \brief policy_adaptor_
    ///
    policy_adaptor_type& policy_adaptor_;

};

template<typename EnvType, typename PolicyType, typename PolicyAdaptorType>
PolicyImprovement<EnvType, PolicyType, PolicyAdaptorType>::PolicyImprovement(real_t gamma, const DynVec<real_t>& val_func,
                                                 policy_type& policy, policy_adaptor_type& policy_adaptor)
    :
      DPAlgoBase<EnvType>(),
      gamma_(gamma),
      v_(val_func),
      policy_(policy),
      policy_adaptor_(policy_adaptor)
{}

template<typename EnvType, typename PolicyType, typename PolicyAdaptorType>
EpisodeInfo
PolicyImprovement<EnvType, PolicyType, PolicyAdaptorType>::on_training_episode(env_type& env, uint_t episode_idx){

    auto start = std::chrono::steady_clock::now();

    std::map<std::string, std::any> options;

    uint_t counter = 0;
    for(uint_t s=0; s<env.n_states(); ++s){

        auto state_actions = state_actions_from_v(env, v_, gamma_, s);

        options.insert_or_assign("state", s);
        options.insert_or_assign("state_actions", std::any(state_actions));
        policy_ = policy_adaptor_(options);
        ++counter;
    }

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<real_t> elapsed_seconds = end-start;

    EpisodeInfo info;
    info.episode_index = episode_idx;
    info.episode_iterations = counter;
    info.total_time = elapsed_seconds;
    return info;
}


}

}

}
}

#endif // POLICY_IMPROVEMENT_H
