#ifndef NEAREST_NEIGHBOR_CONTAINER_H
#define NEAREST_NEIGHBOR_CONTAINER_H

namespace cubeai {
namespace containers{

///
/// \details The NearestNeighborContainer class. Allows
/// for the following queries:
///
/// - Existence: Check if a point is in the container
/// - Nearest Neighbor: Return the closest point (or optionally the closest N points,
/// for any N) to a target point. The target point doesn’t have to be in the container
/// - Region: Return all the points in the container within a certain region, either a
/// hyper-sphere or hyper-rectangle
///
template<typename Data>
class NearestNeighborContainer
{


};
}

}

#endif // NEAREST_NEIGHBOR_CONTAINER_H
