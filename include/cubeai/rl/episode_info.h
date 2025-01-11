#ifndef EPISODE_INFO_H
#define EPISODE_INFO_H

#include "cubeai/base/cubeai_types.h"
#include "rlenvs/rlenvs_consts.h"
#include <chrono>
#include <string>
#include <map>
#include <any>
#include <ostream>

namespace cuberl {
namespace rl {

///
/// \brief The EpisodeInfo struct
///
struct EpisodeInfo
{

    uint_t episode_index{rlenvscpp::consts::INVALID_ID};
    uint_t episode_iterations{rlenvscpp::consts::INVALID_ID};
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
