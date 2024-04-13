// SPDX-FileCopyrightText: 2024 <copyright holder> <email>
// SPDX-License-Identifier: Apache-2.0

#ifndef FIRST_VISIT_MC_H
#define FIRST_VISIT_MC_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_consts.h"
#include "cubeai/rl/episode_info.h"

#include <string>
#include <algorithm>
#include <vector>

namespace cubeai{
namespace rl{
namespace algos {
namespace mc {

struct FirstVisitMCSolverConfig
{
    real_t gamma{1.0};
    real_t tolerance{1.0e-6};
    real_t init_alpha{0.5};
    real_t min_alpha{0.01};
    real_t alpha_decay_ratio{0.3};
    uint_t max_steps{100};
    uint_t n_episodes{500};
    std::string save_path{CubeAIConsts::dummy_string()};
};

/**
 * @todo write docs
 */
template<typename EnvType, typename PolicyType, typename TrajectoryGenerator, typename DecayLRSchedule>
class FirstVisitMCSolver
{
public:

    /**
      * @brief The environment type
      *
      */
    typedef EnvType env_type;

    /**
      * @brief The policy type
      *
      */
    typedef PolicyType policy_type;


    /**
     * @brief The time step type used by the environment
     *
     * */
    typedef typename env_type::time_step_type time_step_type;




    /**
     * @brief Constructor
     *
     **/
    FirstVisitMCSolver(FirstVisitMCSolverConfig solver_config,
                       PolicyType& policy,
                       TrajectoryGenerator& trajectory_gen,
                       DecayLRSchedule& decay_lr_schedule);


    ///
    /// \brief actions_before_training_begins. Execute any actions the
    /// algorithm needs before starting the iterations
    ///
    void actions_before_training_begins(env_type& env);

    ///
    /// \brief actions_after_training_ends. Actions to execute after
    /// the training iterations have finisehd
    ///
    void actions_after_training_ends(env_type& env){}

    ///
    /// \brief actions_before_training_episode
    ///
    void actions_before_episode_begins(env_type&, uint_t /*episode_idx*/){}

    ///
    /// \brief actions_after_training_episode
    ///
    void actions_after_episode_ends(env_type&, uint_t /*episode_idx*/, const EpisodeInfo& /*einfo*/){}

    ///
    /// \brief on_episode Do one on_episode of the algorithm
    ///
    EpisodeInfo on_training_episode(env_type& env, uint_t episode_idx);

    ///
    ///
    ///
    void save(const std::string& filename)const;

private:


    /**
     * @brief The value function
     *
     **/
    DynVec<real_t> v_;

    /**
     * @brief The configuration of the solver
     *
     **/
    FirstVisitMCSolverConfig config_;


    /**
     * @brief The policy to use
     *
     */
    policy_type policy_;


    /**
     * @brief generate the trajector
     *
     */
    TrajectoryGenerator trajectory_gen_;

    /**
     * @brief How to decay the learning rate per episode
     *
     */
    DecayLRSchedule decay_lr_schedule_;

};

template<typename EnvType, typename PolicyType,
         typename TrajectoryGenerator, typename DecayLRSchedule>
FirstVisitMCSolver<EnvType, PolicyType,
                   TrajectoryGenerator, DecayLRSchedule>::FirstVisitMCSolver(FirstVisitMCSolverConfig solver_config,
                                                                             PolicyType& policy,
                                                                             TrajectoryGenerator& trajectory_gen,
                                                                             DecayLRSchedule& decay_lr_schedule)
:
v_(),
policy_(policy),
config_(solver_config),
trajectory_gen_(trajectory_gen),
decay_lr_schedule_(decay_lr_schedule)
{}


template<typename EnvType, typename PolicyType, typename TrajectoryGenerator, typename DecayLRSchedule>
void
FirstVisitMCSolver<EnvType, PolicyType, TrajectoryGenerator, DecayLRSchedule>::actions_before_training_begins(env_type& env){

    v_.resize(env.n_states());
    std::for_each(v_.begin(), v_.end(),
                  [](auto& item){item = 0.0;});
}

template<typename EnvType, typename PolicyType, typename TrajectoryGenerator, typename DecayLRSchedule>
EpisodeInfo
FirstVisitMCSolver<EnvType, PolicyType,
                   TrajectoryGenerator, DecayLRSchedule>::on_training_episode(env_type& env,
                                                                              uint_t episode_idx){


    // generate the trajectory for the environment
    auto trajectory = trajectory_gen_(env, policy_, config_.max_steps);

    // calculate learning rate
    auto alpha = decay_lr_schedule_(config_.init_alpha, episode_idx);

    auto time_step_itr = trajectory.begin();

    for(; time_step_itr != trajectory.end(); ++time_step_itr){

        // find the steps from the current time_step to the end
        // of the trajector
        auto n_steps = std::distance(time_step_itr, trajectory.end());
        auto time_step = *time_step_itr;

        // calculate the return
        auto G = 0.0;

        auto mc_error = G -  v_[time_step.observation()];

        // update the state value
        v_[time_step.observation()] += alpha * mc_error;
    }





}


}
}
}
}
#endif // FIRST_VISIT_MC_H
