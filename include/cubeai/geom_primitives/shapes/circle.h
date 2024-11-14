#ifndef CIRCLE_H
#define CIRCLE_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_consts.h"
#include "cubeai/geom_primitives/geom_point.h"
#include "cubeai/base/math_constants.h"

#include <cmath>

namespace cubeai{
namespace geom_primitives{

///
/// \brief The Circle class. Models a common circle
///
class Circle
{
public:

    ///
    /// \brief Construct a circle centerd at the origin
    ///
    explicit Circle(real_t r);

    ///
    /// \brief Construct a circle given its radius and center
    ///
    Circle(real_t r, const GeomPoint<2>& center);

    ///
    /// \brief Returns the radius of the circle
    ///
    real_t radius()const noexcept{return r_;}

    ///
    /// \brief Returns the center of the circle
    ///
    GeomPoint<2> center()const noexcept{return center_;}

    ///
    /// \brief Returns the area
    ///
    real_t area()const;

    ///
    /// \brief Returns true if the given point lies inside the circle
    ///
    bool is_inside(const GeomPoint<2>& point, real_t tol = CubeAIConsts::tolerance())const;

    /**
     * @brief Returns true if the given spatial point (x, y) lies within the
     * circle.
     *
     */
    bool is_inside(const real_t x, const real_t y, real_t tol = CubeAIConsts::tolerance())const;

private:

    /// \brief The radius of the circle
    real_t r_;


    /// \brief The center of the circle
    GeomPoint<2> center_;

};

inline
Circle::Circle(real_t r)
    :
      r_(r),
      center_(0.0)
{}


inline
Circle::Circle(real_t r, const GeomPoint<2>& center)
    :
   r_(r),
   center_(center)
{}


inline
real_t
Circle::area()const{

    return MathConsts::PI*r_*r_;

}

inline
bool
Circle::is_inside(const GeomPoint<2>& point, real_t tol)const{

    if(std::pow((center_[0] - point[0]), 2) + std::pow((center_[1] - point[1]), 2) - r_*r_ < tol){
        return true;
    }

    return false;

}

inline
bool
Circle::is_inside(const real_t x, const real_t y, real_t tol)const{
    return is_inside({x, y}, tol);
}


}
}
#endif // CIRCLE_H
