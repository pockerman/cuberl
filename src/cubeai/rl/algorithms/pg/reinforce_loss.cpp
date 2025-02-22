#ifndef REINFORCE_LOSS_H
#define REINFORCE_LOSS_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
#include <vector>

namespace cuberl {
namespace rl {
namespace algos {
namespace pg {
	
///
/// \brief compute_loss_item. Compute the product -rewards[i] * log_probs[i]
///
std::vector<torch_tensor_t>
compute_loss_item(const std::vector<real_t>& rewards, 
                  const std::vector<torch_tensor_t>& log_probs){
					  
	
	std::vector<torch_tensor_t> loss_items;
	loss_items.reserve(rewards.size());

	auto rewards_begin = rewards.begin();
	auto rewards_end = rewards.end();
	auto log_probs_begin = log_probs.begin();
	
	for(; rewards_begin != rewards_end; ++rewards_begin, ++log_probs_begin){
		
		auto loss_val = (*rewards_begin) * (*log_probs_begin);
		loss_items.push_back(-loss_val);
	}
					  
	return loss_items;
}

	
	
}
}
}
}


#endif
#endif