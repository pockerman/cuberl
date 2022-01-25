#include "cubeai/rl/rl_mixins.h"

namespace cubeai::rl{

void
with_q_table_mixin::initialize(state_type n_states, action_type n_actions, real_t init_value){

    q_table.resize(n_states, n_actions);


    for(uint_t s=0; s< n_states; ++s){
        for(uint_t a=0; s< n_actions; ++a){
            q_table(s, a) = init_value;
        }
    }
}

void
with_double_q_table_mixin::initialize(state_type n_states, action_type n_actions, real_t init_value){

    q_table_1.resize(n_states, n_actions);
    q_table_2.resize(n_states, n_actions);

    for(uint_t s=0; s< n_states; ++s){
        for(uint_t a=0; s< n_actions; ++a){

            q_table_1(s, a) = init_value;
            q_table_2(s, a) = init_value;
        }
    }
}
}
