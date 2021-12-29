/**
  * A* search on a road network. This example
  * requires that the OSMNX (https://osmnx.readthedocs.io/)
  * library is installed as well as the NetworkX (https://networkx.github.io/documentation/stable/)
  * library. The example is taken from the Autonomous Vehicle course on Coursera
  *
 */


#include "cubeai/base/cubeai_types.h"
//#include "cubeai/planning/a_star_search.h"
#include "cubeai/base/math_constants.h"

#include <boost/python.hpp>

#include <vector>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <limits>
#include <map>
#include <cmath>


namespace example{

typedef boost::python::api::object obj_t;
using cubeai::real_t;
using cubeai::uint_t;

struct Node
{

    uint_t id;
    real_t x;
    real_t y;
    uint_t stret_count;
};

// the heuristic to use for A*
struct DistanceHeuristic
{

    //
    real_t operator()(const Node& n1, const Node& n2)const;

};

real_t
DistanceHeuristic::operator()(const Node& n1, const Node& n2)const{

    auto long1 = n1.x*cubeai::MathConsts::PI/180.0;
    auto lat1  = n1.y*cubeai::MathConsts::PI/180.0;
    auto long2 = n2.x*cubeai::MathConsts::PI/180.0;
    auto lat2  = n2.y*cubeai::MathConsts::PI/180.0;

    // Use a spherical approximation of the earth for
    // estimating the distance between two points.
    auto r = 6371000;
    auto x1 = r*std::cos(lat1)*std::cos(long1);
    auto y1 = r*std::cos(lat1)*std::sin(long1);
    auto z1 = r*std::sin(lat1);

    auto x2 = r*std::cos(lat2)*std::cos(long2);
    auto y2 = r*std::cos(lat2)*std::sin(long2);
    auto z2 = r*std::sin(lat2);

    auto d = (std::pow(x2-x1, 2) + std::pow(y2-y1, 2) + std::pow(z2-z1, 2);
    return std::sqrt(d);
}

// wrapper to facilitate the Python API
class Graph
{
public:

    typedef Node vertex_type;

    // constructor
    Graph(std::string start, std::string end, obj_t py_namespace);

    // build the OSNX graph
    void build();

    uint_t n_nodes()const noexcept{return nodes_.size();}

    void get_path();



private:

    std::string start_;
    std::string end_;
    obj_t py_namespace_;
    obj_t map_graph_;

    Node start_node_;
    Node  end_node_;

    std::vector<Node> nodes_;

    // build the nodes list
    void build_nodes_list_();

};

Graph::Graph(std::string start, std::string end, obj_t py_namespace)
    :
      start_(start),
      end_(end),
      py_namespace_(py_namespace),
      start_node_(),
      end_node_()
{}

void
Graph::build_nodes_list_(){

    std::string nodes_list_str = "node_data = map_graph.nodes(True)\n";
    nodes_list_str += "node_data = list(node_data)\n";
    boost::python::exec(nodes_list_str.c_str(), py_namespace_);

    auto node_data = boost::python::extract<boost::python::list>(py_namespace_["node_data"]);

    nodes_.reserve(boost::python::len(node_data));

    for(auto i=0; i<boost::python::len(node_data); ++i){

        Node n;
        auto node_tuple = boost::python::extract<boost::python::tuple>(node_data()[i]);

        n.id = boost::python::extract<uint_t>(node_tuple()[0]);
        n.x = boost::python::extract<real_t>(node_tuple()[1]["x"]);
        n.y = boost::python::extract<real_t>(node_tuple()[1]["y"]);
        n.stret_count = n.x = boost::python::extract<uint_t>(node_tuple()[1]["street_count"]);
        nodes_.push_back(n);
    }
}

void
Graph::build(){

    std::string imports = "import osmnx as ox\n";
    auto ignore = boost::python::exec(imports.c_str(), py_namespace_);

    std::string map_graph_str = "map_graph = ox.graph_from_place('" + start_ + "," + end_ + "'," + "network_type='drive')\n";

    // fill in the map graph
    boost::python::exec(map_graph_str.c_str(), py_namespace_);

    std::string origin_str = "origin = ox.get_nearest_node(map_graph, (37.8743, -122.277))\n";
    boost::python::exec(origin_str.c_str(), py_namespace_);


    uint_t start_node_id = boost::python::extract<uint_t>(py_namespace_["origin"]);
    std::string destination_str = "destination = list(map_graph.nodes())[-1]";

    boost::python::exec(destination_str.c_str(), py_namespace_);
    uint_t end_node_id = boost::python::extract<uint_t>(py_namespace_["destination"]);

    start_node_.id = start_node_id;
    end_node_.id = end_node_id;

    build_nodes_list_();
}

void
Graph::get_path(){

    DistanceHeuristic dist_h;
    //cubeai::a_star_search(*this, start_node_, end_node_, dist_h);
}

}


int main(){

    using namespace example;

    try{

        Py_Initialize();
        auto main_module = boost::python::import("__main__");
        auto main_namespace = main_module.attr("__dict__");

        // create the graph
        Graph graph("Berkeley", "California", main_namespace);
        graph.build();

        std::cout<<"Number of nodes="<<graph.n_nodes()<<std::endl;

        graph.get_path();
    }
    catch(const boost::python::error_already_set&)
    {
            PyErr_Print();
    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){

        std::cout<<"Unknown exception occured"<<std::endl;
    }
    return 0;
}
