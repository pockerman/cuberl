#ifndef WAYPOINT_PATH_H
#define WAYPOINT_PATH_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/geom_primitives/geom_point.h"
#include "cubeai/geom_primitives/generic_line.h"
#include "cubeai/geom_primitives/geometry_utils.h"


#include <vector>
#include <utility>
#include <stdexcept>

namespace cubeai{
namespace geom_primitives {

/// \brief Helper class to represent a waypoint
template<int dim, typename Data>
struct WayPoint: public GeomPoint<dim>
{
    typedef GeomPoint<dim> position_t;
    typedef Data data_t;
    uint_t id;
    data_t data;
    bool is_active_point;

    ///
    /// constructor
    ///
    WayPoint(const position_t& p, uint_t id_, const data_t& data_=data_t())
        :
          GeomPoint<dim>(p),
          id(id_),
          data(data_),
          is_active_point(true)
    {}

    ///
    /// \brief WayPoint
    /// \param other
    ///
    WayPoint(const WayPoint& other);

    ///
    /// \brief WayPoint
    /// \param other
    ///
    WayPoint(WayPoint&& other);

    ///
    /// \brief Return the id of the point
    ///
    uint_t get_id()const{return id;}

    ///
    /// \brief Writable reference to the data
    ///
    data_t& get_data(){return data;}

    ///
    /// \brief Read reference to the data
    ///
    const data_t& get_data()const{return data;}

    ///
    /// \brief Returns true if the waypoint is active
    ///
    bool is_active()const{return is_active_point;}
};

template<int dim, typename Data>
WayPoint<dim, Data>::WayPoint(const WayPoint<dim, Data>& other)
    :
    GeomPoint<dim>(other),
    id(other.id),
    data(other.data),
    is_active_point(other.is_active_point)
{}

template<int dim, typename Data>
WayPoint<dim, Data>::WayPoint(WayPoint<dim, Data>&& other)
    :
    GeomPoint<dim>(other),
    id(other.id),
    data(std::move(other.data)),
    is_active_point(other.is_active_point)
{}


struct LineSegmentData
{
    ///
    /// \brief The maximum velocity
    /// allowed on the edge
    ///
    real_t vmax{0.0};

    ///
    /// \brief The minimum velocity
    /// allowed on the edge
    ///
    real_t vmin{0.0};

    ///
    /// \brief The maximum velocity
    /// allowed on the edge
    ///
    real_t wmax{0.0};

    ///
    /// \brief The minimum velocity
    /// allowed on the edge
    ///
    real_t wmin{0.0};

    ///
    /// \brief The orientation of the
    /// segment with respect to the global coordinate
    /// frame. This may also dictate the orientation
    /// that a reference vehicle may have on the segment
    ///
    real_t theta{0.0};

    ///
    /// \brief The angular velocity on the segment
    ///
    real_t w{0.0};

    ///
    /// \brief The angular velocity on the segement
    ///
    real_t v{0.0};

};

template<int dim, typename NodeData, typename SegmentData>
class LineSegment: public GenericLine<WayPoint<dim, NodeData>>
{
public:

    static const int dimension = dim;
    typedef NodeData node_data_t;
    typedef SegmentData segment_data_t;
    typedef WayPoint<dim, NodeData> w_point_t;
    //typedef kernel::kernel_detail::generic_line_base<w_point_t> base;
    //typedef typename base::node_type node_type;

    //using base::start;
    //using base::end;
    //using base::get_id;
    //using base::set_id;
    //using base::has_valid_id;

    /// \brief Constructor
    LineSegment(uint_t id, const w_point_t& v1,
                const w_point_t& v2, const segment_data_t& data);


    /// \brief Constructor
    LineSegment(uint_t id, const w_point_t& v1,
                const w_point_t& v2);


    /// \brief Returns the v-th vertex of the segment
    const w_point_t& get_vertex(uint_t v)const;

    /// \brief Returns true if the segment is active
    bool is_active()const{return is_active_;}

    /// \brief deactive the segment
    void deactivate(){is_active_ = false;}

    /// \brief Activate the segment
    void make_active(){is_active_ = true;}

    /// \brief Returns the orientation of the
    /// segment with respect to some global frame
    real_t get_orientation()const{return data_.theta;}

    /// \brief Returns the angular velocity on the
    /// segment
    real_t get_angular_velocity()const{return data_.w;}

    /// \brief Returns the linear velocity on the
    /// segment
    real_t get_velocity()const{return data_.v;}

    /// \brief Returns the Euclidean distance between
    /// the start and end vertices of the segmen
    real_t length()const{return this->end().distance(this->start());}

private:

    /// \brief list of internal points of
    /// the segment.
    std::vector<w_point_t> internal_points_;

    /// \brief The data asociated with the segmen
    segment_data_t data_;

    /// \brief Flag indicating if the segment is active
    bool is_active_;
};

template<int dim, typename NodeData, typename SegmentData>
LineSegment<dim, NodeData, SegmentData>::LineSegment(uint_t id,
                                                     const typename LineSegment<dim, NodeData, SegmentData>::w_point_t& v1,
                                                     const typename LineSegment<dim, NodeData, SegmentData>::w_point_t& v2,
                                                     const typename LineSegment<dim, NodeData, SegmentData>::segment_data_t& data)
    :
     GenericLine<WayPoint<dim, NodeData>>(v1, v2, id),
     internal_points_(),
     data_(data),
     is_active_(true)
{}

template<int dim, typename NodeData, typename SegmentData>
LineSegment<dim, NodeData, SegmentData>::LineSegment(uint_t id,
                                                     const typename LineSegment<dim, NodeData, SegmentData>::w_point_t& v1,
                                                     const typename LineSegment<dim, NodeData, SegmentData>::w_point_t& v2)
    :
    LineSegment<dim, NodeData, SegmentData>(id, v1, v2, typename LineSegment<dim, NodeData, SegmentData>::segment_data_t())
{}


template<int dim, typename NodeData, typename SegmentData>
const typename LineSegment<dim, NodeData, SegmentData>::w_point_t&
LineSegment<dim, NodeData, SegmentData>::get_vertex(uint_t v)const{

    if( v == 0 ){
        return this->start();
    }
    else if(v == 1){
        return this->end();
    }

    throw std::logic_error("Vertex index not in [0,1]");
}

///
/// \brief class WaypointPath models a path formed
/// by line segments and way points. The Data
/// template parameter represents the data held
/// at the waypoints of the path
///
template<int dim, typename PointData, typename EdgeData>
class WaypointPath
{
public:

    static const int dimension = dim;

    typedef PointData w_point_data_t;
    typedef WayPoint<dim, w_point_data_t> w_point_type;
    typedef GeomPoint<dim> point_t;
    typedef EdgeData segment_data_t;
    typedef LineSegment<dim, w_point_data_t, segment_data_t> segment_type;
    typedef segment_type element_t;

    /// \brief point iteration
    typedef typename std::vector<w_point_type>::iterator node_iterator_impl;
    typedef typename std::vector<w_point_type>::const_iterator cnode_iterator_impl;

    /// \brief Line segment  iteration
    typedef typename std::vector<segment_type>::iterator element_iterator_impl;
    typedef typename std::vector<segment_type>::const_iterator celement_iterator_impl;

    ///
    /// \brief Constructor
    ///
    WaypointPath();

    ///
    /// \brief Destructor
    ///
    ~WaypointPath()=default;

    ///
    /// \brief How many waypoints the pah has
    ///
    uint_t n_nodes()const{return waypoints_.size();}

    ///
    /// \brief How many segments the path has
    ///
    //uint_t n_elements()const{return segments_.size();}

    ///
    /// \brief Reserve space for waypoints
    ///
    void reserve_nodes(uint_t n);

    ///
    /// \brief clear the memory allocated for points and
    /// edges
    ///
    void clear();

    ///
    /// \brief front
    /// \return
    ///
    const w_point_type& front()const {return waypoints_.front();}

    ///
    /// \brief front
    /// \return
    ///
    w_point_type& front(){return waypoints_.front();}

    ///
    /// \brief back
    /// \return
    ///
    const w_point_type& back()const {return waypoints_.back();}

    ///
    /// \brief back
    /// \return
    ///
    w_point_type& back(){return waypoints_.back();}

    ///
    /// \brief Add a new waypoint in the path
    /// and get a writable pointer default
    /// waypoint data is assumed
    ///
    w_point_type& add_node(const GeomPoint<dim>& position)
    {return add_node(position, w_point_data_t());}

    ///
    /// \brief Add a new waypoint in the path and get back
    /// a writable reference of the newly added point
    ///
    w_point_type& add_node(const GeomPoint<dim>& position,
                           const w_point_data_t& data);

    /// \brief Raw node iteration
    node_iterator_impl nodes_begin(){return waypoints_.begin();}
    node_iterator_impl nodes_end(){return waypoints_.end();}

    /// \brief Raw node iteration
    cnode_iterator_impl nodes_begin()const{return waypoints_.begin();}
    cnode_iterator_impl nodes_end()const{return waypoints_.end();}



private:

    /// \brief The Waypoints of the path
    std::vector<w_point_type> waypoints_;

};

template<int dim, typename PointData, typename EdgeData>
WaypointPath<dim, PointData, EdgeData>::WaypointPath()
    :
      waypoints_()
      //segments_()
{}

template<int dim, typename PointData, typename EdgeData>
void
WaypointPath<dim, PointData, EdgeData>::clear(){
   waypoints_.clear();
}

template<int dim, typename PointData, typename EdgeData>
typename WaypointPath<dim, PointData, EdgeData>::w_point_type&
WaypointPath<dim, PointData, EdgeData>::add_node(const GeomPoint<dim>& position,
                                                      const typename WaypointPath<dim, PointData, EdgeData>::w_point_data_t& data){

    uint_t id = waypoints_.size();
    //WaypointPath<dim, PointData, EdgeData>::w_point_t* p = new WaypointPath<dim, PointData, EdgeData>::w_point_t(position, id, data);
    waypoints_.push_back(WaypointPath<dim, PointData, EdgeData>::w_point_type(position, id, data));
    return waypoints_[id];
}

/*template<int dim, typename PointData, typename EdgeData>
typename WaypointPath<dim, PointData, EdgeData>::segment_t*
WaypointPath<dim, PointData, EdgeData>::add_element(uint_t vid0, uint_t vid1,
                            const typename WaypointPath<dim, PointData, EdgeData>::segment_data_t& data){

    if(vid0 >= waypoints_.size() ||
            vid1 >= waypoints_.size()){

        throw std::logic_error("Waypoint id given not in range [0, " +
                               std::to_string(waypoints_.size()) + ")");

    }

    if(vid0 == vid1){
       throw std::logic_error("Cannot create segment having identical vertices");
    }

    auto id = segments_.size();
    auto v0 = waypoints_[vid0];
    auto v1 = waypoints_[vid1];
    WaypointPath<dim, PointData, EdgeData>::segment_t* seg = new WaypointPath<dim, PointData, EdgeData>::segment_t(id, *v0, *v1, data);
    segments_.push_back(seg);
    return segments_[id];
}*/

template<int dim, typename PointData, typename EdgeData>
void
WaypointPath<dim, PointData, EdgeData>::reserve_nodes(uint_t n){
    waypoints_.reserve(n);
}

/*template<int dim, typename PointData, typename EdgeData>
void
WaypointPath<dim, PointData, EdgeData>::reserve_elements(uint_t n){
    segments_.reserve(n);
}

template<int dim, typename PointData, typename EdgeData>
typename WaypointPath<dim, PointData, EdgeData>::segment_t*
WaypointPath<dim, PointData, EdgeData>::element(uint_t e){

    if(e >= segments_.size()){
        throw std::logic_error("Invalid segment id: "+
                               std::to_string(e)+
                               "not in [0, "+
                               std::to_string(segments_.size()));
    }

    return segments_[e];
}*/

/*template<int dim, typename PointData, typename EdgeData>
const typename WaypointPath<dim, PointData, EdgeData>::segment_t*
WaypointPath<dim, PointData, EdgeData>::element(uint_t e)const{
    if(e >= segments_.size()){
        throw std::logic_error("Invalid segment id: "+
                               std::to_string(e)+
                               "not in [0, "+
                               std::to_string(segments_.size()));
    }

    return segments_[e];
}*/

template<typename PointData, typename EdgeData>
std::pair<bool, GeomPoint<2>>
compute_projection_of_point_on_path(const WaypointPath<2, PointData, EdgeData>& path, const GeomPoint<2>& point){

    static const uint_t dummy_id = 0;

    auto nodes_begin = path.nodes_begin();
    auto nodes_end = path.nodes_end();

    auto start = *nodes_begin;
    ++nodes_begin;

    for(; nodes_begin != nodes_end; ++nodes_begin){

        auto end = *nodes_begin;
        auto line = LineSegment<2, PointData, EdgeData>(dummy_id, start, end);

        if(geom_utils::point_in_line_vertices(line, point)){
            return {true, geom_utils::compute_projection_of_point_on_line(line, point)};
        }
    }

    return {false, GeomPoint<2>()};
}

///
///
///
template<typename PointData, typename EdgeData>
std::vector<WayPoint<2, PointData>> find_intersections(const WaypointPath<2, PointData, EdgeData>& path,
                                                       const geom_primitives::GeomPoint<2>& position, real_t lookahead_dist){


}

}
}

#endif // WAYPOINT_PATH_H
