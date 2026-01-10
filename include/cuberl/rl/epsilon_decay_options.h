#ifndef EPSILON_DECAY_OPTIONS_H
#define EPSILON_DECAY_OPTIONS_H

#include "cuberl/extern/enum.h"
#include <string>

namespace cuberl {
namespace rl {

///
/// \brief The RenderModeType enum
///
BETTER_ENUM(EpsilonDecayOptionType, char, INVALID_TYPE=0, NONE, EXPONENTIAL, INVERSE_STEP, CONSTANT_RATE);


///
/// \brief to_string.  Returns the RenderModeType to its stringrepresentation
/// \param type The RenderModeType to convert
/// \return std::string

inline
std::string to_string(EpsilonDecayOptionType type){return type._to_string();}


}

}

#endif // EPSILON_DECAY_OPTIONS_H
