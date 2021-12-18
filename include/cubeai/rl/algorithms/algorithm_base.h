#ifndef ALGORITHM_BASE_H
#define ALGORITHM_BASE_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/iterative_algorithm_controller.h"
#include "cubeai/base/iterative_algorithm_result.h"

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
    /// \brief actions_before_training_iterations. Execute any actions the
    /// algorithm needs before starting the iterations
    ///
    virtual void actions_before_training_iterations() = 0;

    ///
    /// \brief actions_after_training_iterations. Actions to execute after
    /// the training iterations have finisehd
    ///
    virtual void actions_after_training_iterations() = 0;

    ///
    /// \brief actions_before_training_episode
    ///
    virtual void actions_before_training_episode(){}

    ///
    /// \brief actions_after_training_episode
    ///
    virtual void actions_after_training_episode(){}

    ///
    /// \brief step Do one step of the algorithm
    ///
    virtual void step() = 0;

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
    /// \brief current_iteration
    /// \return
    ///
    uint_t current_iteration()const{return itr_ctrl_.get_current_iteration();}

    ///
    /// \brief n_max_itrs
    /// \return
    ///
    uint_t n_max_itrs()const{return this->itr_ctrl_.get_max_iterations();}

protected:

    ///
    /// \brief AlgorithmBase
    ///
    AlgorithmBase(uint_t n_max_itrs, real_t tolerance);

    ///
    /// \brief iter_controller
    /// \return
    ///
    IterativeAlgorithmController& iter_controller_(){return itr_ctrl_;}

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
