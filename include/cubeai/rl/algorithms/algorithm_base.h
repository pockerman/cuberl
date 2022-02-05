#ifndef ALGORITHM_BASE_H
#define ALGORITHM_BASE_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/iterative_algorithm_controller.h"
#include "cubeai/base/iterative_algorithm_result.h"
#include "cubeai/rl/algorithms/rl_algo_config.h"
#include <boost/noncopyable.hpp>


namespace cubeai{
namespace rl{
namespace algos {

///
/// \brief The AlgorithmBase class. Base class for deriving
/// RL algorithms
///
class AlgorithmBase: private boost::noncopyable
{

public:

    ///
    /// \brief ~AlgorithmBase. Destructor
    ///
    virtual ~AlgorithmBase() = default;

    ///
    /// \brief actions_before_training_episodes. Execute any actions the
    /// algorithm needs before starting the iterations
    ///
    virtual void actions_before_training_episodes() = 0;

    ///
    /// \brief actions_after_training_episodes. Actions to execute after
    /// the training iterations have finisehd
    ///
    virtual void actions_after_training_episodes() = 0;

    ///
    /// \brief actions_before_training_episode
    ///
    virtual void actions_before_training_episode(){}

    ///
    /// \brief actions_after_training_episode
    ///
    virtual void actions_after_training_episode(){}

    ///
    /// \brief on_episode Do one on_episode of the algorithm
    ///
    virtual void on_episode() = 0;

    ///
    /// \brief reset. Reset the underlying data structures to the point when the constructor is called.
    ///
    virtual void reset();

    ///
    /// \brief train Iterate to train the agent
    ///
    virtual IterativeAlgorithmResult train();

    ///
    /// \brief do_minimla_output
    ///
    void do_minimal_output(){itr_ctrl_.set_show_iterations_flag(false);}

    ///
    /// \brief do_verbose_output
    ///
    void do_verbose_output(){itr_ctrl_.set_show_iterations_flag(true);}

    ///
    /// \brief is_verbose
    /// \return
    ///
    bool is_verbose()const{return itr_ctrl_.show_iterations();}

    ///
    /// \brief render_environment
    /// \return
    ///
    bool render_environment()const noexcept{return render_env_;}

    ///
    /// \brief render_env_frequency
    /// \return
    ///
    uint_t render_env_frequency()const noexcept{return render_env_frequency_;}

    ///
    /// \brief current_episode_idx
    /// \return
    ///
    uint_t current_episode_idx()const noexcept{return itr_ctrl_.get_current_iteration();}

    ///
    /// \brief n_episodes
    /// \return
    ///
    uint_t n_episodes()const noexcept{return this->itr_ctrl_.get_max_iterations();}

    ///
    /// \brief n_iterations_per_episode
    /// \return
    ///
    uint_t n_iterations_per_episode()const noexcept{return n_itrs_per_episode_;}

protected:

    ///
    /// \brief AlgorithmBase
    ///
    AlgorithmBase(uint_t n_episodes, real_t tolerance);


    ///
    /// \brief AlgorithmBase
    /// \param config
    ///
    explicit AlgorithmBase(RLAlgoConfig config);

    ///
    /// \brief iter_controller
    /// \return
    ///
    IterativeAlgorithmController& iter_controller_(){return itr_ctrl_;}

    ///
    /// \brief render_env_
    ///
    bool render_env_;

    ///
    /// \brief render_env_frequency_
    ///
    uint_t render_env_frequency_;

    ///
    /// \brief n_itrs_per_episode_
    ///
    uint_t n_itrs_per_episode_;

private:

    ///
    /// \brief itr_ctrl_. The object controlling the iterations
    ///
    IterativeAlgorithmController itr_ctrl_;

};

}
}
}

#endif // ALGORITHM_BASE_H
