#ifndef DUMMY_ALGORITHM_H
#define DUMMY_ALGORITHM_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/algorithm_base.h"
#include "cubeai/rl/algorithms/rl_algo_config.h"

#include "cubeai/base/cubeai_config.h"

#ifdef USE_GYMFCPP
#include "gymfcpp/render_mode_enum.h"
#endif

#include <iostream>

namespace cubeai{
namespace rl{
namespace algos{

///
///
///
template<typename EnvTp>
class DummyAlgorithm: public AlgorithmBase
{
public:

    ///
    /// \brief env_type
    ///
    typedef EnvTp env_type;

    ///
    /// \brief DummyAlgorithm
    /// \param env
    /// \param config
    ///
    DummyAlgorithm(env_type& env,  RLAlgoConfig config);

    ///
    /// \brief reset
    ///
    virtual void reset() final override;

    ///
    /// \brief actions_before_training_episodes. Execute any actions the
    /// algorithm needs before starting the iterations
    ///
    virtual void actions_before_training_episodes() override final;

    ///
    /// \brief actions_after_training_episodes. Actions to execute after
    /// the training iterations have finisehd
    ///
    virtual void actions_after_training_episodes() override final{}

    ///
    /// \brief on_episode Do one on_episode of the algorithm
    ///
    virtual void on_episode()override final;

protected:

    ///
    /// \brief n_itrs_per_episode_
    ///
    uint_t n_itrs_per_episode_;

    ///
    /// \brief rewards_
    ///
    DynVec<real_t> rewards_;

    ///
    ///
    ///
    env_type& env_;

};

template<typename EnvTp>
DummyAlgorithm<EnvTp>::DummyAlgorithm(env_type& env,  RLAlgoConfig config)
    :
    AlgorithmBase(config),
    n_itrs_per_episode_(config.n_itrs_per_episode),
    rewards_(config.n_episodes, 0.0),
    env_(env)
{}

template<typename EnvTp>
void
DummyAlgorithm<EnvTp>::reset(){

    this->AlgorithmBase::reset();
    env_.reset();
    rewards_ = DynVec<real_t>(rewards_.size(), 0.0);
}

template<typename EnvTp>
void
DummyAlgorithm<EnvTp>::actions_before_training_episodes(){
    this->reset();   
}


template<typename EnvTp>
void
DummyAlgorithm<EnvTp>::on_episode(){

    real_t episode_rewards = 0.0;
    for(uint_t episode_itr=0; episode_itr < n_itrs_per_episode_; ++episode_itr){
        auto action = env_.sample();

        auto time_step = env_.step(action);
        episode_rewards += time_step.reward();

        if((this->render_environment()) && ((episode_itr % this->render_env_frequency()) == 0)){

#ifdef USE_GYMFCPP
           env_.render(gymfcpp::RenderModeType::human);
#endif
        }

        if(time_step.done()){
           break;
        }

     rewards_[this->current_episode_idx()] = episode_rewards;

}

}

}
}
}

#endif // DUMMY_ALGORITHM_H
