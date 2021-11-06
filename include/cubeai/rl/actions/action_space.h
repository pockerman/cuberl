#ifndef ACTION_SPACE_H
#define ACTION_SPACE_H

#include "cubeai/base/cubeai_config.h"
#include "cubeai/base/cubeai_types.h"

#include <string>
#include <vector>

namespace cubeai {
namespace rl {
namespace actions {

struct ActionSpace
{
    std::string type;
    std::vector<uint_t> shape;

    ///
    /// \brief ActionSpace
    ///
    ActionSpace(const std::string& t, std::vector<uint_t>& sh)
        :
          type(t),
          shape(sh)
    {}
};

#ifdef USE_PYTORCH
struct TorchActionSpace
{
    std::string type;
    std::vector<torch_int_t> shape;

    ///
    /// \brief TorchActionSpace
    ///
    TorchActionSpace(const std::string& t, std::vector<torch_int_t>& sh)
        :
          type(t),
          shape(sh)
    {}

};
#endif

}

}

}

#endif // ACTION_SPACE_H
