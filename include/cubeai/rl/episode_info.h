#ifndef EPISODE_INFO_H
#define EPISODE_INFO_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_consts.h"
#include <chrono>
#include <string>
#include <map>
#include <any>
#include <ostream>

namespace cubeai {
namespace rl {

struct EpisodeInfo
{

    uint_t episode_index{CubeAIConsts::INVALID_SIZE_TYPE};
    uint_t episode_iterations{CubeAIConsts::INVALID_SIZE_TYPE};
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

    ///
    /// \brief print
    /// \param out
    /// \return
    ///
    std::ostream& print(std::ostream& out)const noexcept;
};

inline
std::ostream& operator<<(std::ostream& out, const EpisodeInfo& info){
    return info.print(out);
}

}

}

#endif // EPISODE_INFO_H
