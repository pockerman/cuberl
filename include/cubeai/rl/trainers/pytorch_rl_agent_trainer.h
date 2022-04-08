#ifndef PYTORCH_RL_AGENT_TRAINER_H
#define PYTORCH_RL_AGENT_TRAINER_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/iterative_algorithm_result.h"
#include "cubeai/rl/trainers/iterative_rl_trainer_base.h"

#include <torch/torch.h>
#include <boost/noncopyable.hpp>
#include <chrono>
#include <memory>

namespace cubeai {
namespace rl {

struct PyTorchRLTrainerConfig
{
    real_t tolerance;
    uint_t n_episodes;
    uint_t out_msg_frequency;

};

template<typename EnvType, typename AgentType>
class PyTorchRLTrainer final: public IterativeRLTrainerBase<EnvType, AgentType>
{
public:

    typedef EnvType env_type;
    typedef AgentType agent_type;

    ///
    ///
    ///
    PyTorchRLTrainer(const PyTorchRLTrainerConfig config, agent_type& agent, std::unique_ptr<torch::optim::Optimizer> optimizer);


    ///
    /// \brief  actions_after_episode_ends. Execute any actions the algorithm needs after
    /// ending the episode
    ///
    virtual void actions_after_episode_ends(env_type& env, uint_t episode_idx);


protected:


    std::unique_ptr<torch::optim::Optimizer> optimizer_;


};

template<typename EnvType, typename AgentType>
PyTorchRLTrainer<EnvType, AgentType>::PyTorchRLTrainer(const PyTorchRLTrainerConfig config, agent_type& agent,
                                                       std::unique_ptr<torch::optim::Optimizer> optimizer)
    :
    IterativeRLTrainerBase<EnvType, AgentType>(config.n_episodes, config.tolerance, agent, config.out_msg_frequency),
    optimizer_(std::move(optimizer))
{}

template<typename EnvType, typename AgentType>
void
PyTorchRLTrainer<EnvType, AgentType>::actions_after_episode_ends(env_type& env, uint_t episode_idx){

    this->IterativeRLTrainerBase<EnvType, AgentType>::actions_after_episode_ends(env, episode_idx);

    optimizer_->zero_grad();
    // compute the loss
    auto loss = this->agent_.compute_loss();
    loss.backward();
    optimizer_->step();
}

}

}

#endif
#endif // PYTORCH_RL_AGENT_TRAINER_H
