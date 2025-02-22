#include "cubeai/rl/algorithms/utils.h"
#include "cubeai/maths/vector_math.h"
#include <algorithm>

namespace cuberl{
namespace rl{
namespace algos{

std::vector<real_t>
create_discounts_array(real_t base, uint_t npoints){

    std::vector<real_t> points(npoints, 0.0);

    for(uint_t i=0; i<npoints; ++i){
        points[i] = std::pow(base, i);
    }

    return points;
}

std::vector<real_t>
calculate_discounted_return_vector(const std::vector<real_t>& rewards, real_t gamma){
	
	
	std::vector<real_t> returns(rewards.size(), 0.0);
	
	for(uint_t t=0; t<rewards.size(); ++t){
        returns[t] = std::pow(gamma, t)*rewards[t];
    }
	
	return returns;
	
}

real_t
calculate_discounted_return(const std::vector<real_t>& rewards, real_t gamma){
	
	auto discounted_vector = calculate_discounted_return_vector(rewards, gamma);
	return cuberl::maths::sum(discounted_vector);
}

real_t
calculate_mean_discounted_return(const std::vector<real_t>& rewards, real_t gamma){
	auto discounted_vector = calculate_discounted_return_vector(rewards, gamma);
	return cuberl::maths::mean(discounted_vector);
}

std::vector<real_t>
calculate_step_discounted_return(const std::vector<real_t>& rewards, real_t gamma){
	
	std::vector<real_t> discounted_returns(rewards.size());
	
	for(uint_t t=0; t<rewards.size(); ++t){
		
		real_t G = 0.0;
		
		auto begin = rewards.begin();
		
		// advance the iterator t positions
		std::advance(begin, t);
		auto counter = 0;
		for(; begin != rewards.end(); ++begin){
			G += std::pow(gamma, counter++) * (*begin);
		}

		discounted_returns[t] = G;
	}
	
	return discounted_returns;
	
}

//#ifdef USE_PYTORCH
//std::vector<real_t>
//calculate_discounted_returns(torch_tensor_t /*reward*/, torch_tensor_t /*discounts*/, uint_t /*n_workers*/){
//    return std::vector<real_t>();
//}
//#endif

}
}
}
