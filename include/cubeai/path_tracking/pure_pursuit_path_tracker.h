#ifndef PURE_PURSUIT_PATH_TRACKER_H
#define PURE_PURSUIT_PATH_TRACKER_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/geom_primitives/geom_point.h"
#include "cubeai/utils/pose.h"

//#include "kernel/dynamics/system_state.h"
//#include "kernel/geometry/geom_point.h"
//#include "kernel/discretization/line_mesh.h"
//#include "kernel/patterns/observer_base.h"

#include <tuple>
#include "boost/noncopyable.hpp"

namespace cubeai {
namespace control {


///
/// \brief The PurePursuit2DPathTracker class
///
template<typename PathType>
class PurePursuit2DPathTracker: private boost::noncopyable
{

public:

    typedef PathType path_type;

    ///
    /// \brief Constructor
    ///
    PurePursuit2DPathTracker()=default;

    ///
    /// \brief Execute the control
    ///
    std::tuple<geom_primitives::GeomPoint<2>, real_t, int> execute(const utils::Pose<2>& pose);

    ///
    /// \brief Set the lookahead distance
    ///
    void set_lookahead_dist(real_t dist){lookahead_distance_ = dist;}

    ///
    /// \brief Set the goal radius
    ///
    void set_goal_radius(real_t r){goal_radius_ = r;}

    ///
	///
    /// \brief Set the goal position
    ///
    void set_goal(const geom_primitives::GeomPoint<2>& goal){goal_ = goal;}

	///
    /// \brief Set the number of sampling points to
    /// use when computing the closest point from the
    /// position to the path
	///
    void set_n_sampling_points(uint_t npoints){n_sampling_points_ = npoints; }

	///
    /// \brief Update. Notify the observer that the
    /// resource is observing has been changed
	///
    void update(const path_type& path)noexcept{ path_ptr_ = &path;}


private:

    ///
    /// \brief The goal location
    ///
    geom_primitives::GeomPoint<2> goal_;

    /// \brief The lookahed point calculated
    /// by the controller
    ///
    geom_primitives::GeomPoint<2> lookahead_point_;

    ///
    /// \brief The lookahead distance paramter
    ///
    real_t lookahead_distance_{0.0};

    ///
    /// \brief The radius of the circle within which
    /// the goal is assumed to lie
    ///
    real_t goal_radius_{0.0};

    ///
    /// \brief number of sampling poinst to use
    /// when computing the closest point from
    /// position to the path
    ///
    uint_t n_sampling_points_;

    ///
    /// \brief path_ptr_
    ///
    const path_type* path_ptr_{nullptr};
};

template<typename PathType>
std::tuple<geom_primitives::GeomPoint<2>, real_t, int>
PurePursuit2DPathTracker<PathType>::execute(const utils::Pose<2>& pose){

    /// 1. Find the path point closest to the position

    /// get the position coordinates
    real_t rx = pose.position[0];
    real_t ry = pose.position[1];

    /// form the position
    kernel::GeomPoint<2> position({rx, ry});

    const path_t& path=this->read();


    /// find the closest point from the position to the
    /// path
    const kernel::GeomPoint<2> point = kernel::numerics::find_closest_point_to(path, position,
                                                                               n_sampling_points_,
                                                                               kernel::KernelConsts::tolerance());

    /// 2. Find the lookahead point. We can find the lookahead point
    /// by finding the intersection point of the circle centered at
    /// the robot's location and radius equal to the lookahead distance
    /// and the path segment
    auto intersections = kernel::numerics::find_intersections(path,
                                                              kernel::Circle(lookahead_distance_, position));

    if(intersections.empty()){
        /// we cannot proceed
        throw std::logic_error("No intersection points found");
    }

    ///3. calculate the curvature
    /// this will be
    auto theta = state.get("Theta");
    auto tangent =std::tan(theta);
    auto a = -tangent;
    auto b = 1.0;
    auto c = tangent*rx - ry;
    auto lookahead_point = intersections[0];

    auto x = std::fabs(a*lookahead_point[0] +
            b*lookahead_point[1] +c)/std::sqrt(a*a + b*b);

    auto k = 2*x/kernel::utils::sqr(lookahead_distance_*lookahead_distance_);

    /// we need however a signed curvature

    /// construct a point on the robot line
    /*real_t bx = rx + std::cos(theta);
    real_t by = ry + std::sin(theta);*/

    auto sign = kernel::utils::sign(std::sin(theta)*(rx - lookahead_point[0]) -
            std::cos(theta)*(ry - lookahead_point[1]));


    return std::make_tuple(lookahead_point, k, sign);
}

}

}

#endif // PURE_PURSUIT_PATH_TRACKER_H
