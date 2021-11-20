#ifndef VANILLA_REINFORCE_H
#define VANILLA_REINFORCE_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/algorithm_base.h"

namespace cubeai {
namespace rl {
namespace algos {
namespace pg {


struct ReinforceOpts
{
    uint_t max_num_of_episodes;
    uint_t max_itrs_per_episode;
    real_t gamma;
    real_t tolerance;
};


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
    /// \brief actions_before_training_iterations. Execute any actions the
    /// algorithm needs before starting the iterations
    ///
    virtual void actions_before_training_iterations() override;

    ///
    /// \brief actions_after_training_iterations. Actions to execute after
    /// the training iterations have finisehd
    ///
    virtual void actions_after_training_iterations() override{}

    ///
    /// \brief step Do one step of the algorithm
    ///
    virtual void step() override;

private:


    world_t* world_ptr_;
    policy_t policy_ptr_;
    optimizer_t* opt_ptr_;


    uint_t max_itrs_per_episode_;
    real_t gamma_;

};

template<typename WorldTp, typename PolicyTp, typename OptimizerTp>
Reinforce<WorldTp, PolicyTp, OptimizerTp>::Reinforce(ReinforceOpts opts, world_t& world, policy_t& policy, optimizer_t& opt)
    :
     AlgorithmBase(opts.max_num_of_episodes, opts.tolerance),
     world_ptr_(&world),
     policy_ptr_(policy),
     opt_ptr_(&opt),
     max_itrs_per_episode_(opts.max_itrs_per_episode),
     gamma_(opts.gamma)
{}

template<typename WorldTp, typename PolicyTp, typename OptimizerTp>
void
Reinforce<WorldTp, PolicyTp, OptimizerTp>::actions_before_training_iterations(){
}

template<typename WorldTp, typename PolicyTp, typename OptimizerTp>
void Reinforce<WorldTp, PolicyTp, OptimizerTp>::step(){


    //  for every episode reset the environment
    auto state = world_ptr_ ->reset();
    // self._reset_internal_structs()

    for(uint_t itr=0; itr < max_itrs_per_episode_; ++itr){

            auto [action, log_prob] = policy_ptr_ -> act(state);

            //self.saved_log_probs.append(log_prob)
            auto time_step = world_ptr_ ->step(action);
            //self.rewards.append(reward)

            if (time_step.done()){
                break;
            }
    }

    //self.scores_deque.append(sum(self.rewards))
    // self.scores.append(sum(self.rewards))

    //discounts = [self.gamma ** i for i in range(len(self.rewards) + 1)]
    //R = sum([a * b for a, b in zip(discounts, self.rewards)])

    std::vector<real_t> policy_loss;
    //for log_prob in self.saved_log_probs:
    //        policy_loss.append(-log_prob * R)
    policy_ptr_ -> update_policy_loss(policy_loss);
    //auto policy_loss = torch.cat(policy_loss).sum()

    opt_ptr_ -> zero_grad();
    policy_ptr_ -> step_backward_policy_loss();
    //policy_loss.backward();
    opt_ptr_ -> step();

    //current_episode_idx = self.current_itr_index
    //if current_episode_idx  % self.print_frequency == 0:
    //print('Episode {}\tAverage Score: {:.2f}'.format(current_episode_idx, np.mean(self.scores_deque)))
    //if np.mean(self.scores_deque) >= 195.0:
    //        print('Environment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(current_episode_idx - 100,
    //                                                                                   np.mean(self.scores_deque)))
    //        self.itr_control.residual = self.itr_control.residual * 10**-2




}

}
}
}
}

#endif // VANILLA_REINFORCE_H
