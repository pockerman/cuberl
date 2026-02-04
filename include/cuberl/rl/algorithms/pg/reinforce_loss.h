#ifndef REINFORCE_LOSS_H
#define REINFORCE_LOSS_H

#include "cuberl/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cuberl/base/cuberl_types.h"
#include "bitrl/bitrl_consts.h"

#include <vector>

namespace cuberl {
namespace rl {
namespace algos {
namespace pg {
	
	//using namespace bitrl::consts;
	
///
/// \brief compute_loss_item. Compute the product -rewards[i] * log_probs[i]
///
std::vector<torch_tensor_t>
compute_loss_item(const std::vector<real_t>& rewards, 
                  const std::vector<torch_tensor_t>& log_probs);

///
/// \brief Compute the baseline using a user-defined constant
/// value from the given rewards 
///
std::vector<real_t> 
compute_baseline_with_constant(const std::vector<real_t>& rewards,
							   real_t constant);
							   
							   
///
/// \brief Compute the baseline using the mean value of the 
/// provided rewards
///
std::vector<real_t> 
compute_baseline_with_mean(const std::vector<real_t>& rewards);
						   
///
/// \brief Compute the baseline using standarization also known as
/// whitening
///
std::vector<real_t> 
compute_baseline_with_standardization(const std::vector<real_t>& rewards, 
                                      real_t eps=bitrl::consts::TOLERANCE);
	
}
}
}
}


#endif
#endif