/*
#ifndef VANILLA_REINFORCE_H
#define VANILLA_REINFORCE_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/algorithm_base.h"
#include "gymfcpp/render_mode_enum.h"

#include <vector>
#include <deque>
#include <numeric>
#include <iostream>

namespace cubeai {
namespace rl {
namespace algos {
namespace pg {


///
/// \brief The ReinforceOpts struct. Holds various
/// configuration options for the Reinforce algorithm
///
struct ReinforceOpts
{
    uint_t max_num_of_episodes;
    uint_t max_itrs_per_episode;
    uint_t print_frequency;
    uint_t scores_queue_max_size;
    real_t gamma;
    real_t tolerance;
    real_t exit_score_level;
    bool render_environment;

    ///
    /// \brief print
    /// \param out
    /// \return
    ///
    std::ostream& print(std::ostream& out)const;
};

inline
std::ostream& operator<<(std::ostream& out, ReinforceOpts opts){
    return opts.print(out);
}

///
/// \brief The Reinfoce class. Vanilla Reinforce algorithm
///
template<typename WorldTp, typename PolicyTp, typename OptimizerTp>
class Reinforce final: public AlgorithmBase
{
public:

    typedef WorldTp world_t;
    typedef PolicyTp policy_t;
    typedef OptimizerTp optimizer_t;

    ///
    /// \brief Reinforce
    ///
    Reinforce(ReinforceOpts opts, world_t& world, policy_t& policy, optimizer_t& opt);

    ///
    /// \brief actions_before_training_episodes. Execute any actions the
    /// algorithm needs before starting the iterations
    ///
    virtual void actions_before_training_episodes() override;

    ///
    /// \brief actions_after_training_episodes. Actions to execute after
    /// the training iterations have finisehd
    ///
    virtual void actions_after_training_episodes() override{}

    ///
    /// \brief on_episode Do one on_episode of the algorithm
    ///
    virtual void on_episode() override;

    ///
    /// \brief print_frequency
    ///
    uint_t print_frequency()const noexcept{return print_frequency_;}

private:

    ///
    /// \brief world_ptr_
    ///
    world_t* world_ptr_;

    ///
    /// \brief policy_ptr_
    ///
    policy_t policy_ptr_;

    ///
    /// \brief opt_ptr_
    ///
    optimizer_t* opt_ptr_;

    ///
    /// \brief max_itrs_per_episode_
    ///
    uint_t max_itrs_per_episode_;

    ///
    /// \brief print_frequency_
    ///
    uint_t print_frequency_;

    ///
    /// \brief scores_queue_max_size_
    ///
    uint_t scores_queue_max_size_;

    ///
    /// \brief gamma_
    ///
    real_t gamma_;

    ///
    /// \brief exit_score_level_
    ///
    real_t exit_score_level_;

    ///
    /// \brief render_environment_
    ///
    bool render_environment_;

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
    void do_step_();

};

template<typename WorldTp, typename PolicyTp, typename OptimizerTp>
Reinforce<WorldTp, PolicyTp, OptimizerTp>::Reinforce(ReinforceOpts opts, world_t& world, policy_t& policy, optimizer_t& opt)
    :
     AlgorithmBase(opts.max_num_of_episodes, opts.tolerance),
     world_ptr_(&world),
     policy_ptr_(policy),
     opt_ptr_(&opt),
     max_itrs_per_episode_(opts.max_itrs_per_episode),
     print_frequency_(opts.print_frequency),
     scores_queue_max_size_(opts.scores_queue_max_size),
     gamma_(opts.gamma),
     exit_score_level_(opts.exit_score_level),
     render_environment_(opts.render_environment)
{}

template<typename WorldTp, typename PolicyTp, typename OptimizerTp>
void
Reinforce<WorldTp, PolicyTp, OptimizerTp>::actions_before_training_episodes(){

    world_ptr_-> reset();
    scores_.clear();
    reset_internal_structs_();

}

template<typename WorldTp, typename PolicyTp, typename OptimizerTp>
void
Reinforce<WorldTp, PolicyTp, OptimizerTp>::reset_internal_structs_()noexcept{

    std::vector<real_t> empty;
    std::swap(saved_log_probs_, empty);
    empty.clear();
    std::swap(rewards_, empty);

    std::deque<real_t> empty_deque;
    std::swap(scores_deque_, empty_deque);

}

template<typename WorldTp, typename PolicyTp, typename OptimizerTp>
void
Reinforce<WorldTp, PolicyTp, OptimizerTp>::compute_discounts_(std::vector<real_t>& data)const noexcept{

    for(uint_t i=0; i < rewards_.size() + 1; ++i ){
        data.push_back(std::pow(gamma_, i));
    }
}

template<typename WorldTp, typename PolicyTp, typename OptimizerTp>
real_t
Reinforce<WorldTp, PolicyTp, OptimizerTp>::compute_total_reward_(const std::vector<real_t>& discounts)const{

    real_t reward = 0.0;

    for(uint_t i = 0; i<rewards_.size(); ++i){
        reward += discounts[i]*rewards_[i];
    }

    return reward;
}

template<typename WorldTp, typename PolicyTp, typename OptimizerTp>
void Reinforce<WorldTp, PolicyTp, OptimizerTp>::do_step_(){

    //  for every episode reset the environment
    auto state = world_ptr_ ->reset().observation();

    for(uint_t itr=0; itr < max_itrs_per_episode_; ++itr){

            auto [action, log_prob] = policy_ptr_ -> act(state);

             if(render_environment_){
                world_ptr_->render(gymfcpp::RenderModeType::human);
            }

            saved_log_probs_.push_back(log_prob);
            auto time_step = world_ptr_ ->step(action);
            state = time_step.observation();

            rewards_.push_back(time_step.reward());

            if (time_step.done()){
                break;
            }
    }

}

template<typename WorldTp, typename PolicyTp, typename OptimizerTp>
void Reinforce<WorldTp, PolicyTp, OptimizerTp>::on_episode(){

    //  for every episode reset the environment
    //auto state = world_ptr_ ->reset().observation();
    reset_internal_structs_();

    do_step_();

    auto rewards_sum = std::accumulate(rewards_.begin(), rewards_.end(), 0.0);

    // remove oldest element if needed
    if(scores_deque_.size() >= scores_queue_max_size_){
        scores_deque_.pop_front();
    }

    scores_deque_.push_back(rewards_sum);
    scores_.push_back(rewards_sum);

    std::vector<real_t> discounts;
    discounts.reserve(rewards_.size() + 1);

    //discounts = [self.gamma ** i for i in range(len(self.rewards) + 1)]
    compute_discounts_(discounts);

    // R = sum([a * b for a, b in zip(discounts, self.rewards)])
    auto R = compute_total_reward_(discounts);

    std::vector<real_t> policy_loss_values;
    policy_loss_values.reserve(saved_log_probs_.size());

    for(auto& log_prob: saved_log_probs_){
        policy_loss_values.push_back(-log_prob * R);
    }


    policy_ptr_ -> update_policy_loss(policy_loss_values);
    opt_ptr_ -> zero_grad();

    // backward propagate policy loss i.e. policy_loss.backward();
    policy_ptr_ -> step_backward_policy_loss();
    opt_ptr_ -> step();

    auto scores_mean = std::accumulate(scores_deque_.begin(), scores_deque_.end(), 0.0);
    scores_mean /= std::distance(scores_deque_.begin(), scores_deque_.end());


    if(this->current_episode_idx() % print_frequency() == 0){
        std::cout<<"Episode: "<<this->current_episode_idx()<<" average score: "<<scores_mean<<std::endl;
    }

    if(scores_mean > exit_score_level_){
        std::cout<<"Environment solved in "<<this->current_episode_idx()<<". Average score: "<<scores_mean<<std::endl;
        auto res = this->iter_controller_().get_residual();
        this->iter_controller_().update_residual(res * 1.0e-2);
    }
}

}
}
}
}

#endif // VANILLA_REINFORCE_H


*/
