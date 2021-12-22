#ifndef RL_ALGO_INPUT_H
#define RL_ALGO_INPUT_H

#include "cubeai/base/cubeai_types.h"

namespace cubeai{
namespace rl{
namespace algos{


struct RLAlgoConfig
{
    uint_t n_episodes;
    uint_t n_itrs_per_episode;
    real_t tolerance;
    bool render_environment{false};
    uint_t render_env_frequency;
    uint_t output_msg_frequency;
};
}
}
}

#endif // RL_ALGO_INPUT_H
