/**
  * Example 0: Demonstrates the use of the DummyAlgorithm class.
  * This class exposes the basic API that most implemented RL
  * algorithms expose.
  *
  * */

#include "cubeai/base/cubeai_types.h"
#include "cubeai/estimation/kalman_filter.h"
#include "cubeai/utils/iteration_counter.h"

#include <iostream>
#include <unordered_map>

namespace example_0
{



using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynMat;
using cubeai::DynVec;
using cubeai::estimation::KalmanFilterConfig;
using cubeai::estimation::KalmanFilter;
using cubeai::utils::IterationCounter;


}

int main() {

    using namespace example_0;

    try{

    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){

        std::cout<<"Unknown exception occured"<<std::endl;
    }

   return 0;
}
