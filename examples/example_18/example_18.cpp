/**
  * Double Q-learning on CartPole-v0 environment
  *
 */


#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/math_constants.h"

#include "gymfcpp/cart_pole.h"

#include <boost/python.hpp>

#include <vector>
#include <map>
#include <queue>
#include <set>
#include <iostream>
#include <limits>
#include <cmath>
#include <exception>
#include <algorithm>


namespace example{

typedef boost::python::api::object obj_t;
using cubeai::real_t;
using cubeai::uint_t;


}


int main(){

    using namespace example;

    try{

        Py_Initialize();
        auto main_module = boost::python::import("__main__");
        auto main_namespace = main_module.attr("__dict__");

        std::cout<<__cplusplus<<std::endl;

#if __cplusplus >= 202002L
        std::cout<<"Using C++20 standard"<<std::endl;
#else
        std::cout<<"Using C++17 standard"<<std::endl;
#endif
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
