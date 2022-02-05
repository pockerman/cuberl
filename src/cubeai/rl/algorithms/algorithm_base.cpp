#include "cubeai/rl/algorithms/algorithm_base.h"
#include "cubeai/base/iterative_algorithm_controller.h"

#include <iostream>

namespace cubeai{
namespace rl {
namespace algos {

AlgorithmBase::AlgorithmBase(uint_t n_episodes, real_t tolerance)
    :
    itr_ctrl_(n_episodes, tolerance)
{}

AlgorithmBase::AlgorithmBase(RLAlgoConfig config)
    :
    AlgorithmBase(config.n_episodes, config.tolerance)

{
    render_env_ = config.render_environment;
    render_env_frequency_ = config.render_env_frequency;
    n_itrs_per_episode_ = config.n_itrs_per_episode;
}


IterativeAlgorithmResult
AlgorithmBase::train(){

    this->actions_before_training_episodes();

    while(itr_ctrl_.continue_iterations()){

        if(itr_ctrl_.show_iterations()){
            std::cout<<"Iteration="<<itr_ctrl_.get_current_iteration()<<" of "<<itr_ctrl_.get_max_iterations()<<std::endl;
            std::cout<<"Residual="<<itr_ctrl_.get_residual()<<" Exit tolerance= "<<itr_ctrl_.get_exit_tolerance()<<std::endl;
        }

        actions_before_training_episode();
        this->on_episode();
        actions_after_training_episode();
    }

    this->actions_after_training_episodes();

    return itr_ctrl_.get_state();

}

void
AlgorithmBase::reset(){
    itr_ctrl_.reset();
}

}

}
}
