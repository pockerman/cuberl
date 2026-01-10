#ifndef OPTIMIZER_TYPE_H
#define OPTIMIZER_TYPE_H

#include "cuberl/extern/enum.h"
#include <string>

namespace cuberl {
namespace maths{
namespace optim{

///
/// \brief The RenderModeType enum
///
BETTER_ENUM(OptimzerType, char, INVALID_TYPE=-1, GD=0, SGD, ADAM, RSPROP);

///
/// \brief to_string.  Returns the RenderModeType to its stringrepresentation
/// \param type The RenderModeType to convert
/// \return std::string

inline
std::string to_string(OptimzerType type){return type._to_string();}

inline
constexpr bool compare(OptimzerType tp1, OptimzerType tp2){
    return tp1 == tp2;
}

}

}
}
#endif // OPTIMIZER_TYPE_H
