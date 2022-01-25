#ifndef RL_MIXINS_H
#define RL_MIXINS_H
/**
  * Utility class used in the implementation
  * of RL algorithms
  * */
#include "cubeai/base/cubeai_types.h"


namespace cubeai::rl{

///
/// \brief The WithQTableMixin struct
///
struct with_q_table_mixin
{
    typedef uint_t state_type;
    typedef uint_t action_type;
    typedef real_t value_type;

    ///
    /// \brief q_table
    ///
    DynMat<value_type> q_table;

    ///
    /// \brief initialize
    /// \param n_states
    /// \param n_actions
    /// \param init_value
    ///
    void initialize(state_type n_states, action_type n_actions, real_t init_value);
};

///
/// \brief The WithDoubleQTableMixin struct
///
struct with_double_q_table_mixin
{
    typedef uint_t state_type;
    typedef uint_t action_type;
    typedef real_t value_type ;

    ///
    /// \brief q_table
    ///
    DynMat<value_type> q_table_1;

    ///
    /// \brief q_table_2
    ///
    DynMat<value_type> q_table_2;

    ///
    /// \brief initialize
    /// \param n_states
    /// \param n_actions
    /// \param init_value
    ///
    void initialize(state_type n_states, action_type n_actions, real_t init_value);
};



}

#endif // RL_MIXINS_H
