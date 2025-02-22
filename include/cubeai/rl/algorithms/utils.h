#ifndef UTILS_H
#define UTILS_H

///
/// \brief Various utilities used when working with RL
/// problems
///

#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_config.h"
#include "cubeai/maths/vector_math.h"

#include <vector>
#include <iostream>

namespace cuberl{
namespace rl {
namespace algos {

///
/// \brief Given the state index returns the list of actions under the
/// provided value functions
///
template<typename WorldTp>
auto state_actions_from_v(const WorldTp& env, 
                          const DynVec<real_t>& v,
                          real_t gamma, 
						  uint_t state) -> DynVec<real_t>{

    auto q = DynVec<real_t>(env.n_actions());
    std::for_each(q.begin(),
                  q.end(),
                  [](auto& item){item = 0.0;});
    

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
/// \brief Create a vector where element i  is the product
///  $$\gamma^i * rewards[i]$$
///
std::vector<real_t>
calculate_discounted_return_vector(const std::vector<real_t>& rewards, 
                                   real_t gamma);

///
/// \brief calculate_discounted_return_vector. Creates the discounted return vector
/// for the given trajectory
///
template<typename TimeStepType>
std::vector<real_t>
calculate_discounted_return_vector(const std::vector<TimeStepType>& trajectory, 
                                   real_t gamma){
									   
	std::vector<real_t> rewards(trajectory.size());
	for(uint_t t =0; t<trajectory.size(); ++t){
		rewards[t] = std::pow(gamma, t)*trajectory[t].reward(); 
	}
	
	return rewards;
	
}

///
/// \brief calculate_discounted_return. Calculates the sum of the discounted
/// rewards for the given rewards array using the given gamma
/// \param rewards
/// \param gamma
/// \return
///
real_t
calculate_discounted_return(const std::vector<real_t>& rewards, real_t gamma);

///
/// \brief calculate_mean_discounted_return. Same as calculate_discounted_return
/// but the result is weighted by 1/N where N is the size of the given rewards array
/// \param rewards
/// \param gamma
/// \return
///
real_t
calculate_mean_discounted_return(const std::vector<real_t>& rewards, real_t gamma);

///
/// \brief Calculate the discounted return from the given trajectory
///
template<typename TimeStepType>
real_t 
calculate_discounted_return(const std::vector<TimeStepType>& trajectory, real_t gamma){

	auto discounted_vector = calculate_discounted_return_vector(trajectory, gamma);
	return cuberl::maths::sum(discounted_vector);
}

template<typename TimeStepType>
real_t
calculate_mean_discounted_return(const std::vector<TimeStepType>& trajectory, 
                                 real_t gamma){
									 
	auto discounted_vector = calculate_discounted_return_vector(trajectory, gamma);
	return cuberl::maths::mean(discounted_vector);
	
}

///
/// \brief Given an array of rewards, for each entry calculate the
/// following:
/// $$G = \sum_{k=t+1}^T \gamma^{k-t-1}R_k$$
///
std::vector<real_t>
calculate_step_discounted_return(const std::vector<real_t>& rewards, real_t gamma);

//#ifdef USE_PYTORCH
/////
///// \brief calculate_discounted_returns
///// \param reward
///// \param discounts
///// \param n_workers
///// \return
/////
//std::vector<real_t>
//calculate_discounted_returns(torch_tensor_t reward, torch_tensor_t discounts, uint_t n_workers);
//#endif
}
}
}

#endif // UTILS_H
