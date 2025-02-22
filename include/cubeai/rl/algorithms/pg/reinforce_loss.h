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
                  const std::vector<torch_tensor_t>& log_probs);

	
	
}
}
}
}


#endif
#endif