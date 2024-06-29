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
#include "cubeai/utils/torch_adaptor.h"
#include "cubeai/rl/algorithms/rl_algorithm_base.h"
#include "cubeai/rl/algorithms/utils.h"
#include "cubeai/rl/episode_info.h"
#include "cubeai/utils/cubeai_concepts.h"
#include "cubeai/utils/torch_adaptor.h"
#include "cubeai/data_structs/experience_buffer.h"

#ifdef CUBEAI_DEBUG
#include <cassert>
#endif

#include <string>
#include <chrono>
#include <map>
#include <any>
#include <memory>
#include <tuple>
#include <string>

namespace cubeai{
namespace rl{
namespace algos {
namespace ac {

namespace  {


// utility functio to stack the values under prop_name
// in the given buffer as PyTorch tensor.
// TODO: maybe we need to guard against the prop_name
template<typename ExperienceType>
torch_tensor_t
stack_values(const cubeai::containers::ExperienceBuffer<ExperienceType>& buffer,
             const std::string& prop_name ){

    std::vector<torch_tensor_t> vals;
    vals.reserve(buffer.size());

    for(const auto& item: buffer){
        auto& info = std::get<4>(item);

        auto value_itr = info.find(prop_name);
        auto values = std::any_cast<torch_tensor_t>(value_itr -> second);
        vals.push_back(values);
    }

    return torch::cat(vals);
}

}

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
    uint_t buffer_size{100};

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

/**
 * @brief A2C solver assuming separate networks for the actor and
 * the critic. In additio the implementation assumes that the
 * given environment handles any multiplicity of the environments
 *
 */
template<typename EnvType, typename PolicyType, typename CriticType>
class A2CSolver final: public RLSolverBase<EnvType>
{
public:

    /**
     * @brief The environment type
     */
    typedef EnvType env_type;

    /**
     * @brief The  policy or action type
     */
    typedef PolicyType policy_type;

    /**
     * @brief The critic type
     */
    typedef CriticType critic_type;

    ///
    /// \brief A2C
    /// \param config
    /// \param policy
    ///
    A2CSolver(const A2CConfig config,
              policy_type& policy, critic_type& critic,
              std::unique_ptr<torch::optim::Optimizer>& policy_optimizer,
              std::unique_ptr<torch::optim::Optimizer>& critic_optimizer);

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
    virtual void actions_after_episode_ends(env_type&, uint_t /*episode_idx*/, const EpisodeInfo& info);

    ///
    /// \brief on_episode Do one on_episode of the algorithm
    ///
    virtual EpisodeInfo on_training_episode(env_type&, uint_t /*episode_idx*/);

    ///
    /// \brief set_train_mode for both the Actor and the Critic
    ///
    void set_train_mode()noexcept;

    ///
    /// \brief set_evaluation_mode for both the Actor and the Critic
    ///
    void set_evaluation_mode()noexcept;

private:

    // helper class
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
    policy_type& policy_;

    /**
     * @brief The action network
     */
    critic_type& critic_;

    /**
     * @brief The policy_ optimzer
     */
    std::unique_ptr<torch::optim::Optimizer> policy_optimizer_;

    /**
     * @brief The optimizer for the critic network
     */
    std::unique_ptr<torch::optim::Optimizer> critic_optimizer_;

    ///
    ///
    ///
    EpisodeInfo do_train_on_episode_(env_type&, uint_t /*episode_idx*/);

    ///
    /// \brief act_on_episode_iteration_ For every episode iteration
    /// this function computes the result for the critic and actor networks
    /// \param state
    ///
    action_result act_on_episode_iteration_(torch_tensor_t& state);

    ///
    /// \brief optimize_model_
    /// \param logprobs
    /// \param entropies
    /// \param values
    /// \param rewards
    ///
    //torch_tensor_t compute_loss_(torch_tensor_t logprobs, torch_tensor_t entropies,
    //                             torch_tensor_t values, torch_tensor_t rewards);

};

template<typename EnvType, typename PolicyType, typename CriticType>
A2CSolver<EnvType, PolicyType, CriticType>::A2CSolver(const A2CConfig config,
                                                      policy_type& policy, critic_type& critic,
                                                      std::unique_ptr<torch::optim::Optimizer>& policy_optimizer,
                                                      std::unique_ptr<torch::optim::Optimizer>& critic_optimizer)
    :
      config_(config),
      policy_(policy),
      critic_(critic),
      policy_optimizer_(std::move(policy_optimizer)),
      critic_optimizer_(std::move(critic_optimizer))
{}

template<typename EnvType, typename PolicyType, typename CriticType>
void
A2CSolver<EnvType, PolicyType, CriticType>::set_train_mode()noexcept{
    policy_ -> train();
    critic_ -> train();

}

template<typename EnvType, typename PolicyType, typename CriticType>
void
A2CSolver<EnvType, PolicyType, CriticType>::set_evaluation_mode()noexcept{
    policy_ -> eval();
    critic_ -> eval();

}

template<typename EnvType, typename PolicyType, typename CriticType>
void
A2CSolver<EnvType, PolicyType, CriticType>::actions_before_training_begins(env_type& env){
    set_train_mode();
}

template<typename EnvType, typename PolicyType, typename CriticType>
EpisodeInfo
A2CSolver<EnvType, PolicyType, CriticType>::on_training_episode(env_type& env, uint_t episode_idx){

    auto start = std::chrono::steady_clock::now();

    uint_t itrs = 0;
    auto R = 0.0;

    auto eps_info = do_train_on_episode_(env, episode_idx);

    auto end = std::chrono::steady_clock::now();
    std::chrono::duration<real_t> elapsed_seconds = end - start;

    EpisodeInfo info;
    info.episode_index = episode_idx;
    info.episode_reward = R;
    info.episode_iterations = eps_info.episode_iterations;
    info.total_time = elapsed_seconds;
    return info;
}

template<typename EnvType, typename PolicyType, typename CriticType>
EpisodeInfo
A2CSolver<EnvType, PolicyType, CriticType>::do_train_on_episode_(env_type& env, uint_t episode_idx){

    auto episode_score = 0.0;

    typedef torch_utils::TorchAdaptor::state_type state_type;
    typedef torch_utils::TorchAdaptor::value_type value_type;
    typedef typename EnvType::time_step_type time_step_type;

    typedef std::tuple<state_type,
                       state_type,
                       real_t,
                       time_step_type,
                       std::map<std::string, std::any>> experience_type;

    // create a buffer for experience accummulation
    cubeai::containers::ExperienceBuffer<experience_type> buffer(config_.buffer_size);
    auto time_step = env.reset();
    auto state = time_step.observation();

    // helper class to convert from std::vector to torch_tensor_t
    // and vice versa
    torch_utils::TorchAdaptor torch_adaptor;

    // loop over the iterations
    uint_t itrs = 0;
    for(; itrs < config_.n_iterations_per_episode; ++ itrs){

        auto torch_state = torch_adaptor(state);
        auto action_result = act_on_episode_iteration_(torch_state);

        auto action = torch_utils::TorchAdaptor::to_vector<uint_t>(action_result.actions)[0];
        auto next_time_step = env.step(action);

        auto next_state = next_time_step.observation();
        auto reward = next_time_step.reward();

        std::map<std::string, std::any> info;
        info["log_probs"] = action_result.log_probs;
        info["values"] = action_result.values;

        // how do we handle the fact that the environment exited?
        // do not append anything in the buffer exit the loop

        auto torch_next_state = torch_adaptor(next_state);
        buffer.append(std::make_tuple(torch_state,
                                      torch_next_state,
                                      reward,
                                      next_time_step,
                                      info));

        state = next_state;

        if(buffer.size() < config_.buffer_size){
            continue;
        }
    }

    // optimize the model check the actions_after_episode_ends
    // function for this

    EpisodeInfo info;
    info.episode_index = episode_idx;
    info.episode_reward = episode_score;
    info.episode_iterations = itrs;
    info.info["log_probs"] = stack_values(buffer, "log_probs"); //stack_log_probs(buffer);
    info.info["values"] = stack_values(buffer, "values");
    info.info["rewards"] = stack_values(buffer, "rewards"); //stack_rewards(buffer);
    return info;
}

template<typename EnvType, typename PolicyType, typename CriticType>
typename A2CSolver<EnvType, PolicyType, CriticType>::action_result
A2CSolver<EnvType, PolicyType, CriticType>::act_on_episode_iteration_(torch_tensor_t& state){

    // get the probabilities and the values
    auto probs = policy_ -> forward(state);
    auto values = critic_ -> forward(state);

    // sample the actions
    auto actions = policy_ -> sample();
    auto logprobs = policy_ -> log_probabilities(actions);

    typedef typename A2CSolver<EnvType, PolicyType, CriticType>::action_result action_result;

    action_result result;
    result.actions = actions;
    result.log_probs = logprobs;
    result.values = values;
    return result;
}

template<typename EnvType,typename PolicyType, typename CriticType>
void
A2CSolver<EnvType, PolicyType, CriticType>::actions_after_episode_ends(env_type&, uint_t /*episode_idx*/,
                                                                       const EpisodeInfo& info){

        auto logprobs_itr = info.info.find("log_probs");
        auto values_itr = info.info.find("values");
        auto rewards_itr = info.info.find("rewards");

        if(logprobs_itr == info.info.end() || values_itr == info.info.end() ||
            rewards_itr == info.info.end()){
            throw std::logic_error("Couldn't find needed item");
        }

        auto logprobs = std::any_cast<torch_tensor_t>(logprobs_itr -> second);
        auto values = std::any_cast<torch_tensor_t>(values_itr -> second);
        auto rewards = std::any_cast<torch_tensor_t>(rewards_itr -> second);

        auto advantage = rewards - values;

        auto policy_loss = -(logprobs * advantage.detach()).mean();
        auto critic_loss = advantage.pow(2).mean();

        // compute the loss we have two networks so two losses
        // auto loss = compute_loss_(std::any_cast<torch_tensor_t>(logprobs->second), torch_tensor_t(),
        //                          std::any_cast<torch_tensor_t>(values->second),
        //                          std::any_cast<torch_tensor_t>(rewards->second));

        // Backward pass and optimize
        policy_optimizer_->zero_grad();
        critic_optimizer_ -> zero_grad();

        policy_loss.backward();
        critic_loss.backward();

        policy_optimizer_ -> step();
        critic_optimizer_ -> step();


}


/*
template<typename EnvType, typename PolicyType, typename CriticType>
torch_tensor_t
A2CSolver<EnvType, PolicyType, CriticType>::compute_loss_(torch_tensor_t logprobs, torch_tensor_t entropies,
                                                          torch_tensor_t values, torch_tensor_t rewards){


        // create the discounts
        auto discounts = create_discounts_array(config_.gamma, rewards.size(0));

        auto vec_rewards = torch_utils::TorchAdaptor::to_vector<real_t>(rewards);
        auto vec_values = torch_utils::TorchAdaptor::to_vector<real_t>(values.detach());

        // calculate the discounted returns
        auto discounted_returns = calculate_discounted_returns(vec_rewards, discounts, config_.n_workers);

        // compute the advanatges
        auto advantages = compute_advantages_(vec_rewards, vec_values);

        // form the loss function

        // clip the grad if needed
        torch::nn::utils::clip_grad_norm_(parameters(), config_.max_grad_norm);

        # get the discounted returns
        discounted_returns: np.array = calculate_discounted_returns(rewards.numpy(),
                                                                    discounts,
                                                                    n_workers=self.config.n_workers)

        advantages: np.array = self._compute_advantages(rewards=rewards.numpy(),
                                                        values=values.detach().numpy())

        loss: torch.Tensor = self._compute_loss_function(advantages=torch.from_numpy(advantages), values=values,
                                                         entropies=entropies,
                                                         returns=torch.from_numpy(discounted_returns),
                                                         logprobs=logprobs[:-1])


        loss.backward()

        # clip the grad if needed
        torch.nn.utils.clip_grad_norm_(self.parameters(),
                                       self.config.max_grad_norm)


}

*/

}

}
}
}
#endif
#endif // A2C_H
