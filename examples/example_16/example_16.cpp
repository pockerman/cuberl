/**
  * Simple implementation of Monte Carlo tree serach
  * algorithm on the Taxi environment. The code below is a translation of
  * the Python code in
  * https://github.com/ashishrana160796/prototyping-self-driving-agents/blob/master/milestone-four/monte_carlo_tree_search_taxi_v3.ipynb
  *
 */


#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/algorithm_base.h"
#include "cubeai/utils/array_utils.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/taxi.h"
#include "gymfcpp/time_step.h"

#include <boost/python.hpp>

#include <vector>
#include <iostream>
#include <unordered_map>
#include <utility>


namespace example{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::algos::AlgorithmBase;
using cubeai::rl::VectorExperienceBuffer;
using gymfcpp::Taxi;

typedef Taxi::time_step_t time_step_t;


const real_t GAMMA = 1.0;
const uint_t N_EPISODES = 20000;
const uint_t N_ITRS_PER_EPISODE = 2000;
const real_t TOL = 1.0e-8;


template<typename Env>
class ApproxMC: public AlgorithmBase
{
public:


};










}


int main(){

    using namespace example;

    try{

        Py_Initialize();
        auto main_module = boost::python::import("__main__");
        auto main_namespace = main_module.attr("__dict__");

        Taxi env("v0", main_namespace, false);
        env.make();

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
