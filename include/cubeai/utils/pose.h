#ifndef POSE_H
#define POSE_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/geom_primitives/geom_point.h"

#include <ostream>

namespace cubeai {
namespace utils {



template<int dim> struct Pose;

template<>
struct Pose<2>
{
    geom_primitives::GeomPoint<2> position;
    real_t orientation;

};

inline
std::ostream& operator<<(std::ostream& out, const Pose<2>& pose){
    out<<"x="<<pose.position[0]<<" y="<<pose.position[1]<<" theta="<<pose.orientation<<std::endl;
    return out;
}

template<>
struct Pose<3>
{
    geom_primitives::GeomPoint<3> position;
    DynVec<real_t> orientation;

};

inline
std::ostream& operator<<(std::ostream& out, const Pose<3>& pose){
    out<<"x="<<pose.position[0]<<" y="<<pose.position[1]<<" z="<<pose.position[2]<<std::endl;
    out<<"phi="<<pose.orientation[0]<<" theta="<<pose.orientation[1]<<" psi="<<pose.orientation[2]<<std::endl;
    return out;
}


}
}
#endif // POSE_H
