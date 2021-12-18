#ifndef NUMPY_UTILS_H
#define NUMPY_UTILS_H

#include "cubeai/base/cubeai_types.h"
#include <vector>

namespace cubeai{
namespace numpy_utils{

///
/// \brief linespace
/// \param a
/// \param b
/// \param n
/// \return
///
std::vector<real_t> linespace(real_t a, real_t b, uint_t n);

}

}

#endif // NUMPY_UTILS_H
