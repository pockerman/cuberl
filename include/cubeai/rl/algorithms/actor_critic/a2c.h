#ifndef A2C_H
#define A2C_H

/**
  * Implements synchronous shared-weights advantage-actor critic, A2C, algorithm
  * Currently the implementation of this class assumes that
  * PyTorch is used to model the deep networks
  *
  */

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/rl_algorithm_base.h"
#include "cubeai/rl/episode_info.h"
#include "cubeai/utils/cubeai_concepts.h"
#include "cubeai/utils/torch_adaptor.h"

#ifdef CUBEAI_DEBUG
#include <cassert>
#endif

#include <string>
#include <chrono>

namespace cubeai{
namespace rl{
namespace algos {
namespace ac {

///
/// \brief The A2CConfig struct. Configuration for A2C class
///
struct A2CConfig
{

    ///
    /// \brief Discount factor
    ///
    real_t gamma{0.99};

    ///
    /// \brief GAE lambda
    ///
    real_t lambda{0.1};

    ///
    /// \brief Coefficient for accounting for entropy contribution
    ///
    real_t beta{0.0};

    ///
    /// \brief policy_loss_weight. How much weight to give
    /// on the policy loss when forming the global loss
    ///
    real_t policy_loss_weight{ 1.0};

    ///
    ///
    ///
    real_t value_loss_weight{1.0};

    ///
    ///
    ///
    real_t max_grad_norm{1.0};

    ///
    ///
    ///
    uint_t n_iterations_per_episode{100};

    ///
    ///
    ///
    uint_t n_workers{1};

    ///
    ///
    ///
    uint_t batch_size{1};

    ///
    ///
    ///
    bool normalize_advantages{true};

    ///
    ///
    ///
    std::string device{"cpu"};

    ///
    ///
    ///
    std::string save_model_path{""};
};


template<typename EnvType, utils::concepts::pytorch_module PolicyType>
class A2C final: public RLAlgoBase<EnvType>
{
public:

    typedef EnvType env_type;
    typedef PolicyType policy_type;

    ///
    /// \brief A2C
    /// \param config
    /// \param policy
    ///
    A2C(const A2CConfig config,  policy_type& policy);

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
    virtual void actions_after_episode_ends(env_type&, uint_t /*episode_idx*/){}

    ///
    /// \brief on_episode Do one on_episode of the algorithm
    ///
    virtual EpisodeInfo on_training_episode(env_type&, uint_t /*episode_idx*/);

    ///
    ///
    ///
    std::vector<torch::Tensor> parameters(bool recurse = true) const{return policy_ -> parameters(recurse);}

private:

    struct action_result
    {
        torch_tensor_t log_probs;
        torch_tensor_t values;
        torch_tensor_t actions;
        torch_tensor_t entropies;

    };

    ///
    /// \brief config_
    ///
    A2CConfig config_;

    ///
    /// \brief policy_
    ///
    policy_type policy_;

    ///
    ///
    ///
    EpisodeInfo do_train_on_episode_(env_type&, uint_t /*episode_idx*/);

    ///
    /// \brief act_on_iteration_
    /// \param state
    ///
    action_result act_on_iteration_(torch_tensor_t& state);

};

template<typename EnvType, utils::concepts::pytorch_module PolicyType>
A2C<EnvType, PolicyType>::A2C(const A2CConfig config,  policy_type& policy)
    :
      config_(config),
      policy_(policy)
{}

template<typename EnvType, utils::concepts::pytorch_module PolicyType>
void
A2C<EnvType, PolicyType>::actions_before_training_begins(env_type& env){

#ifdef CUBEAI_DEBUG
    assert(env.n_copies() == config_.n_workers && "Invalid number of workers");
#endif

}

template<typename EnvType, utils::concepts::pytorch_module PolicyType>
EpisodeInfo
A2C<EnvType, PolicyType>::on_training_episode(env_type& env, uint_t episode_idx){

    auto start = std::chrono::steady_clock::now();

    uint_t itrs = 0;
    auto R = 0.0;

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<real_t> elapsed_seconds = end - start;

    do_train_on_episode_(env, episode_idx);

    EpisodeInfo info;
    info.episode_index = episode_idx;
    info.episode_reward = R;
    info.episode_iterations = itrs;
    info.total_time = elapsed_seconds;
    return info;
}

template<typename EnvType, utils::concepts::pytorch_module PolicyType>
EpisodeInfo
A2C<EnvType, PolicyType>::do_train_on_episode_(env_type& env, uint_t episode_idx){

    auto episode_score = 0.0;

    // this is in parallel all
    // participating workers reset their
    // environment and return their TimeStep
    auto time_step = env.reset();
    auto states = time_step.template stack_states<torch_utils::TorchAdaptor>();  //observation(); //stack_observations();

    // loop over the iterations
    uint_t itrs = 0;
    for(; itrs < config_.n_iterations_per_episode; ++ itrs){

        auto action_result = act_on_iteration_(states);
        auto next_state = env.step(torch_utils::TorchAdaptor::to_vector<uint_t>(action_result.actions));
    }

    EpisodeInfo info;
    info.episode_index = episode_idx;
    info.episode_reward = episode_score;
    info.episode_iterations = itrs;
    return info;
}

template<typename EnvType, utils::concepts::pytorch_module PolicyType>
typename A2C<EnvType, PolicyType>::action_result
A2C<EnvType, PolicyType>::act_on_iteration_(torch_tensor_t& state){

    // get the logits and the values
    auto[logits, values] = policy_.forward(state);

    // sample the actions
    auto actions = policy_.sample(logits);
    auto logprobs = policy_.log_probabilities(actions);

    typedef typename A2C<EnvType, PolicyType>::action_result action_result;

    action_result result;
    result.actions = actions;
    result.log_probs = logprobs;
    result.values = values;
    return result;
}

}

}

}
}
#endif
#endif // A2C_H
