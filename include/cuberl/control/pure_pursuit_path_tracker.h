#ifndef PURE_PURSUIT_PATH_TRACKER_H
#define PURE_PURSUIT_PATH_TRACKER_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/geom_primitives/geom_point.h"
#include "cubeai/geom_primitives/geometry_utils.h"
#include "cubeai/geom_primitives/waypoint_path.h"
#include "cubeai/utils/pose.h"

#include "boost/noncopyable.hpp"
#include <tuple>

namespace cubeai {
namespace control {


///
/// \brief The PurePursuit2DPathTracker class
///
class PurePursuit2DPathTracker: private boost::noncopyable
{

public:

    ///
    /// \brief Constructor
    ///
    PurePursuit2DPathTracker()=default;

    ///
    /// \brief PurePursuit2DPathTracker
    /// \param goal
    /// \param lookahead_dist
    /// \param goal_radius
    ///
    PurePursuit2DPathTracker(const geom_primitives::GeomPoint<2>& goal,
                             real_t lookahead_dist, real_t goal_radius);

    ///
    /// \brief Execute the control
    ///
    template<typename PointData, typename EdgeData>
    std::tuple<geom_primitives::GeomPoint<2>, real_t, int>
    execute(const geom_primitives::WaypointPath<2, PointData, EdgeData>& path,
            const utils::Pose<2>& pose)const;

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

};

inline
PurePursuit2DPathTracker::PurePursuit2DPathTracker(const geom_primitives::GeomPoint<2>& goal,
                                                   real_t lookahead_dist, real_t goal_radius)
    :
      goal_(goal),
      lookahead_point_(),
      lookahead_distance_(lookahead_dist),
      goal_radius_(goal_radius)
{}

template<typename PointData, typename EdgeData>
std::tuple<geom_primitives::GeomPoint<2>, real_t, int>
PurePursuit2DPathTracker::execute(const geom_primitives::WaypointPath<2, PointData, EdgeData>& path,
                                  const utils::Pose<2>& pose)const{

    // find the closest point on the
    // path using the position
    // 1. Find the path point closest to the position

    // get the position coordinates
    real_t rx = pose.position[0];
    real_t ry = pose.position[1];

    // form the position
    geom_primitives::GeomPoint<2> position({rx, ry});

    auto point = geom_primitives::compute_projection_of_point_on_path(path, position);

    // 2. Find the lookahead point. We can find the lookahead point
    // by finding the intersection point of the circle centered at
    // the robot's location and radius equal to the lookahead distance
    // and the path segment
    /*auto intersections = kernel::numerics::find_intersections(path,
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
    //real_t bx = rx + std::cos(theta);
    //real_t by = ry + std::sin(theta);

    auto sign = kernel::utils::sign(std::sin(theta)*(rx - lookahead_point[0]) -
            std::cos(theta)*(ry - lookahead_point[1]));


    return std::make_tuple(lookahead_point, k, sign);*/

}

/*template<typename PathType>
std::tuple<geom_primitives::GeomPoint<2>, real_t, int>
PurePursuit2DPathTracker<PathType>::execute(const utils::Pose<2>& pose){

    /// 1. Find the path point closest to the position

    /// get the position coordinates
    real_t rx = pose.position[0];
    real_t ry = pose.position[1];

    /// form the position
    kernel::GeomPoint<2> position({rx, ry});

    const path_t& path=this->read();


    // find the closest point from the position to the path
    const  point = geom_utils::compute_projection_of_point_on_line(path, position);

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
    //real_t bx = rx + std::cos(theta);
    //real_t by = ry + std::sin(theta);

    auto sign = kernel::utils::sign(std::sin(theta)*(rx - lookahead_point[0]) -
            std::cos(theta)*(ry - lookahead_point[1]));


    return std::make_tuple(lookahead_point, k, sign);
}*/

}

}

#endif // PURE_PURSUIT_PATH_TRACKER_H
