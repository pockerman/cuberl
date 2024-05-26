#ifndef SIMPLE_REINFORCE_H
#define SIMPLE_REINFORCE_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/rl_algorithm_base.h"
#include "cubeai/rl/episode_info.h"
#include "cubeai/maths/vector_math.h"
//#include "cubeai/utils/array_utils.h"


#include <vector>
#include <deque>
#include <numeric>
#include <iostream>
#include <chrono>

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
template<typename EnvType, typename PolicyType>
class ReinforceSolver final: public RLSolverBase<EnvType>
{
public:

    typedef EnvType env_type;
    typedef PolicyType policy_type;

    ///
    /// \brief Reinforce
    ///
    ReinforceSolver(ReinforceConfig opts, policy_type& policy);

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

    ///
    ///
    ///
    std::vector<torch::Tensor> parameters(bool recurse = true) const{return policy_ptr_ -> parameters(recurse);}


    ///
    /// \brief compute_loss
    /// \return
    ///
    torch_tensor_t compute_loss(){return policy_ptr_->compute_loss();}


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
    /// \brief exit_score_level_
    ///
    real_t exit_score_level_;


    std::vector<real_t> scores_;
    std::deque<real_t> scores_deque_;
    std::vector<real_t> saved_log_probs_;
    std::vector<real_t> rewards_;

    ///
    /// \brief reset_internal_structs_
    ///
    void reset_internal_structs_()noexcept;

    ///
    /// \brief compute_discounts
    ///
    void compute_discounts_(std::vector<real_t>& data)const noexcept;

    ///
    /// \brief comoute_rewards_
    ///
    real_t compute_total_reward_(const std::vector<real_t>& discounts)const;

    ///
    /// \brief do_step_
    ///
    uint_t do_step_(env_type& env);

};

template<typename EnvType, typename PolicyType>
ReinforceSolver<EnvType, PolicyType>::ReinforceSolver(ReinforceConfig config, policy_type& policy)
    :
     RLSolverBase<EnvType>(),
     config_(config),
     policy_ptr_(policy)

{}

template<typename EnvType, typename PolicyType>
void
ReinforceSolver<EnvType, PolicyType>::actions_before_training_begins(env_type& /*env*/){

    scores_.clear();
    reset_internal_structs_();

}

template<typename EnvType, typename PolicyType>
void
ReinforceSolver<EnvType, PolicyType>::reset_internal_structs_()noexcept{

    std::vector<real_t> empty;
    std::swap(saved_log_probs_, empty);
    empty.clear();
    std::swap(rewards_, empty);

    std::deque<real_t> empty_deque;
    std::swap(scores_deque_, empty_deque);

}

/*template<typename WorldTp, typename PolicyTp, typename OptimizerTp>
void
Reinforce<WorldTp, PolicyTp, OptimizerTp>::compute_discounts_(std::vector<real_t>& data)const noexcept{

    for(uint_t i=0; i < rewards_.size() + 1; ++i ){
        data.push_back(std::pow(gamma_, i));
    }
}*/

template<typename EnvType, typename PolicyType>
real_t
ReinforceSolver<EnvType, PolicyType>::compute_total_reward_(const std::vector<real_t>& discounts)const{

    real_t reward = 0.0;

    for(uint_t i = 0; i<rewards_.size(); ++i){
        reward += discounts[i]*rewards_[i];
    }

    return reward;
}

template<typename EnvType, typename PolicyType>
uint_t
ReinforceSolver<EnvType, PolicyType>::do_step_(env_type& env){

    //  for every episode reset the environment
    auto state = env.reset().observation();

    uint_t itr = 0;
    for(; itr < config_.max_itrs_per_episode; ++itr){

            // from the policy get the action to do based
            // on the seen state
            auto [action, log_prob] = policy_ptr_ -> act(state);

            saved_log_probs_.push_back(log_prob);

            // execute the selected action on the environment
            auto time_step = env.step(action);

            // update state
            state = time_step.observation();

            rewards_.push_back(time_step.reward());

            if (time_step.done()){
                break;
            }
    }

    return itr;

}

template<typename EnvType, typename PolicyType>
EpisodeInfo
ReinforceSolver<EnvType, PolicyType>::on_training_episode(env_type& env, uint_t episode_idx){


    auto start = std::chrono::steady_clock::now();

    //  for every episode reset the environment
    //auto state = world_ptr_ ->reset().observation();
    reset_internal_structs_();

    auto itrs = do_step_(env);
    auto rewards_sum = std::accumulate(rewards_.begin(), rewards_.end(), 0.0);

    scores_deque_.push_back(rewards_sum);
    scores_.push_back(rewards_sum);

    std::vector<real_t> discounts;
    discounts.reserve(rewards_.size() + 1);

    //discounts = [self.gamma ** i for i in range(len(self.rewards) + 1)]
    cubeai::maths::exponentiate(discounts);

    // R = sum([a * b for a, b in zip(discounts, self.rewards)])
    auto R = compute_total_reward_(discounts);

    std::vector<real_t> policy_loss_values;
    policy_loss_values.reserve(saved_log_probs_.size());

    for(auto& log_prob: saved_log_probs_){
        policy_loss_values.push_back(-log_prob * R);
    }


    policy_ptr_ -> update_policy_loss(policy_loss_values);
    //opt_ptr_ -> zero_grad();

    // backward propagate policy loss i.e. policy_loss.backward();
    //policy_ptr_ -> step_backward_policy_loss();
    //opt_ptr_ -> step();

    //auto scores_mean = std::accumulate(scores_deque_.begin(), scores_deque_.end(), 0.0);
    //scores_mean /= std::distance(scores_deque_.begin(), scores_deque_.end());

    /*if(scores_mean > exit_score_level_){
        std::cout<<"Environment solved in "<<this->current_episode_idx()<<". Average score: "<<scores_mean<<std::endl;
        auto res = this->iter_controller_().get_residual();
        this->iter_controller_().update_residual(res * 1.0e-2);
    }*/

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

template<typename EnvType, typename PolicyType>
void
ReinforceSolver<EnvType, PolicyType>::actions_after_episode_ends(env_type&, uint_t /*episode_idx*/,
                                                               const EpisodeInfo& /*einfo*/){

    // compute the loss
    auto loss = compute_loss();
    loss.backward();


}

}
}
}
}
#endif
#endif // VANILLA_REINFORCE_H
