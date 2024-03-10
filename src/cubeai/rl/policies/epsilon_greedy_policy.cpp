#include "cubeai/rl/policies/epsilon_greedy_policy.h"

#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_config.h"

#ifdef CUBEAI_DEBUG
#include <cassert>
#endif

#include <vector>

namespace cubeai {
namespace rl {
namespace policies {


template<>
uint_t
EpsilonGreedyPolicy::operator()(const DynMat<real_t>& mat, uint_t state_idx)const{

#ifdef CUBEAI_DEBUG
    assert(state_idx < mat.size() && "Invalid state index. Should be state_idx < q_map.size()");
#endif

    return (*this)(mat.row(state_idx));
}

template<>
uint_t
EpsilonGreedyPolicy::operator()(const std::vector<std::vector<real_t>>& mat,
                                uint_t state_idx)const{
#ifdef CUBEAI_DEBUG
    assert(state_idx < mat.size() && "Invalid state index. Should be state_idx < q_map.size()");
#endif

    return (*this)(mat[state_idx]);

}

void
EpsilonGreedyPolicy::on_episode(uint_t episode)noexcept{

    if(decay_op_ == EpsilonDecayOption::NONE)
        return;

    if(decay_op_  == EpsilonDecayOption::INVERSE_STEP ){

        if(episode == 0){
            episode = 1;
        }

        // there are various methods to do epsilon
        // reduction
        eps_ = 1.0 / episode;

        if(eps_ < min_eps_){
            eps_ = min_eps_;
        }
    }
    else if(decay_op_ == EpsilonDecayOption::EXPONENTIAL){
        eps_ = min_eps_ + (max_eps_ - min_eps_)*std::exp(-epsilon_decay_ * episode);
    }
    else if(decay_op_ == EpsilonDecayOption::CONSTANT_RATE){

        eps_ -= epsilon_decay_;

        if(eps_ < min_eps_){
            eps_ = min_eps_;
        }

    }
}

}
}
}
