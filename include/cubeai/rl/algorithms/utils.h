#ifndef UTILS_H
#define UTILS_H

#include "cubeai/base/cubeai_types.h"

#include "cubeai/base/cubeai_config.h"



#include <vector>
#include <iostream>

namespace cubeai{
namespace rl {
namespace algos {

///
/// Given the state index returns the list of actions under the
/// provided value functions
///
template<typename WorldTp>
auto state_actions_from_v(const WorldTp& env, const DynVec<real_t>& v,
                          real_t gamma, uint_t state) -> DynVec<real_t>{

    auto q = DynVec<real_t>(env.n_actions());
    std::for_each(q.begin(),
                  q.end(),
                  [](auto& item){item = 0.0;});
    //auto q = DynVec<real_t>(env.n_actions(), 0.0);

    for(uint_t a=0; a < env.n_actions(); ++a){

        const auto& transition_dyn = env.p(state, a);

        for(auto& dyn: transition_dyn){
            auto prob = std::get<0>(dyn);
            auto next_state = std::get<1>(dyn);
            auto reward = std::get<2>(dyn);
            //auto done = std::get<3>(dyn);
            q[a] += prob * (reward + gamma * v[next_state]);
        }
    }

    return q;
}

///
/// \brief create_discounts_array
/// \param end
/// \param base
/// \param start
/// \param endpoint
/// \return
///
std::vector<real_t> create_discounts_array(real_t base, uint_t npoints);

///
/// \brief calculate_discounted_returns
/// \param rewards
/// \param discounts
/// \param n_workers
/// \return
///
std::vector<real_t> calculate_discounted_returns(const std::vector<real_t>& rewards,
                                                 const std::vector<real_t>& discounts, uint_t n_workers);

#ifdef USE_PYTORCH
///
/// \brief calculate_discounted_returns
/// \param reward
/// \param discounts
/// \param n_workers
/// \return
///
std::vector<real_t>
calculate_discounted_returns(torch_tensor_t reward, torch_tensor_t discounts, uint_t n_workers);
#endif
}
}
}

#endif // UTILS_H
