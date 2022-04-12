#ifndef POSE_H
#define POSE_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/geom_primitives/geom_point.h"

namespace cubeai {
namespace utils {



template<int dim> struct Pose;

template<>
struct Pose<2>
{
    geom_primitives::GeomPoint<2> position;
    real_t orientation;

};


template<>
struct Pose<3>
{
    geom_primitives::GeomPoint<3> position;
    DynVec<real_t> orientation;

};


}
}
#endif // POSE_H
