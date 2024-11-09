#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH


#include "cubeai/rl/algorithms/pg/simple_reinforce.h"

namespace cubeai {
namespace rl {
namespace algos {
namespace pg {

std::ostream&
ReinforceOpts::print(std::ostream& out)const{

    out<<"Max num episodes=    "<<max_num_of_episodes<<std::endl;
    out<<"Max its per episode= "<<max_itrs_per_episode<<std::endl;
    out<<"Print frequency=     "<<print_frequency<<std::endl;
    out<<"Score queue maz size="<<scores_queue_max_size<<std::endl;
    out<<"Gamma=               "<<gamma<<std::endl;
    out<<"Exit tolerance=      "<<tolerance<<std::endl;
    out<<"Exit score level=    "<<exit_score_level<<std::endl;
    return out;
}

}
}
}
}

#endif

