#ifndef RL_MIXINS_H
#define RL_MIXINS_H
/**
  * Utility class used in the implementation
  * of RL algorithms
  * */
#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/epsilon_decay_options.h"

#include <map>


namespace cubeai::rl{

///
/// \brief max_action
/// \param qtable
/// \param state
/// \param n_actions
/// \return
///
uint_t max_action(const DynMat<real_t>& qtable, uint_t state, uint_t n_actions);

struct with_decay_epslion_option_mixin
{
    real_t eps_init;
    real_t eps;
    real_t min_eps;
    real_t max_eps;
    real_t epsilon_decay;
    uint_t n_actions;
    uint_t seed;
    EpsilonDecayOptionType decay_op;

    ///
    /// \brief decay_eps
    /// \param episode_index
    /// \return
    ///
    real_t decay_eps(uint_t episode_index);
};

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

template<typename TableTp>
struct with_double_q_table_mixin;

///
/// \brief The WithDoubleQTableMixin struct
///
template<>
struct with_double_q_table_mixin< DynMat<real_t> >
{
    typedef uint_t index_type;
    typedef uint_t state_type;
    typedef uint_t action_type;
    typedef real_t value_type ;

    ///
    /// \brief q_table_1
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
    void initialize(const std::vector<index_type>& indices, action_type n_actions, real_t init_value);
};

template<typename KeyTp>
struct with_double_q_table_mixin<std::map<KeyTp, DynVec<real_t>>>
{

    typedef KeyTp index_type;
    typedef KeyTp state_type;
    typedef uint_t action_type;
    typedef real_t value_type ;

    ///
    /// \brief q_table_1
    ///
    std::map<KeyTp, DynVec<real_t>> q_table_1;

    ///
    /// \brief q_table_2
    ///
    std::map<KeyTp, DynVec<real_t>> q_table_2;

    ///
    /// \brief initialize
    /// \param n_states
    /// \param n_actions
    /// \param init_value
    ///
    void initialize(const std::vector<index_type>& indices, action_type n_actions, real_t init_value);

};

template<typename KeyTp>
void
with_double_q_table_mixin<std::map<KeyTp, DynVec<real_t>>>::initialize(const std::vector<index_type>& indices, action_type n_actions, real_t init_value){


    DynVec<real_t> init_vals(n_actions, init_value);

    for(uint_t i=0; i< indices.size(); ++i){

        q_table_1[indices[i]] = init_vals;
        q_table_2[indices[i]] = init_vals;
    }
}



}

#endif // RL_MIXINS_H
