#ifndef BOUNDING_BOX_TYPE_H
#define BOUNDING_BOX_TYPE_H

#include "cubeai/extern/enum.h"
#include<string>

namespace cubeai{
namespace geom_primitives{

///
/// \brief The RenderModeType enum
///
BETTER_ENUM(BoundingBoxType, char, RECTANGLE=0, CIRCLE, SPHERE, INVALID_TYPE);


///
/// \brief to_string.  Returns the BoundingBoxType to its string representation
/// \param type The BoundingBoxType to convert
/// \return std::string
inline
std::string to_string(BoundingBoxType type){return type._to_string();}

}
}

#endif // BOUNDING_BOX_TYPE_H
