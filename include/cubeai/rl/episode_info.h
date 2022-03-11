#ifndef EPISODE_INFO_H
#define EPISODE_INFO_H

#include "cubeai/base/cubeai_types.h"
#include <chrono>
#include <string>
#include <map>
#include <any>


namespace cubeai {
namespace rl {

struct EpisodeInfo
{

    uint_t episode_index;
    uint_t episode_iterations{0};
    bool stop_training{false};

    ///
    ///
    ///
    real_t episode_reward{0.0};

    ///
    /// \brief total_time
    ///
    std::chrono::duration<real_t> total_time;

    ///
    /// \brief info
    ///
    std::map<std::string, std::any> info;
};

}

}

#endif // EPISODE_INFO_H
