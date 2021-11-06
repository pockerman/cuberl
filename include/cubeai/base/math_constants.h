#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "cubeai/base/cubeai_types.h"
#include "boost/noncopyable.hpp"

namespace cubeai{

///
/// \brief MathConsts
///
class MathConsts: private boost::noncopyable
{
public:

    ///
    /// \brief Constructor
    ///
    MathConsts() = delete;

    ///
    /// \brief The mathematical constant PI
    ///
    constexpr static real_t PI = 3.14159265359;


private:


};

}

#endif // CONSTANTS_H
