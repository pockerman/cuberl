#ifndef ITERATION_MIXIN_H
#define ITERATION_MIXIN_H

#include "cubeai/base/cubeai_types.h"

namespace cubeai {

///
/// \brief The IterationMixin struct
///
class IterationMixin
{
public:

    ///
    /// \brief Constructor
    ///
    explicit IterationMixin(uint_t maxIterations, bool show_iters=false);

    ///
    /// \brief Destructor
    ///
    ~IterationMixin()=default;

    ///
    /// \brief Returns true if the iterations of the algorithm should be continued
    ///
    bool continue_iterations() noexcept;

    ///
    /// \brief show iterations
    ///
    bool show_iterations()const noexcept{return show_iterations_;}

    ///
    /// \brief show iterations
    ///
    void set_show_iterations_flag(bool flag)noexcept{show_iterations_ = flag;}

    ///
    /// \brief set_max_itrs
    /// \param max_itrs
    ///
    void set_max_itrs(uint_t max_itrs)noexcept{max_iterations_ = max_itrs;}

    ///
    /// \brief Returns the current iteration index
    ///
    uint_t get_current_iteration()const noexcept{return current_iteration_idx_;}

    ///
    /// \brief Return the maximum number of iterations
    ///
    uint_t get_max_iterations()const noexcept{return max_iterations_;}

    ///
    /// \brief reset
    ///
    void reset()noexcept{current_iteration_idx_ = 0; }

private:

    uint_t max_iterations_;
    uint_t current_iteration_idx_;
    bool show_iterations_;

};

inline
IterationMixin::IterationMixin(uint_t maxIterations, bool show_iters)
    :
    max_iterations_(maxIterations),
    current_iteration_idx_(0),
    show_iterations_(show_iters)
{}


}

#endif // ITERATION_MIXIN_H
