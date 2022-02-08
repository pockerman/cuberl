/**
  * Example 0: Demonstrates the use of the DummyAlgorithm class.
  * This class exposes the basic API that most implemented RL
  * algorithms expose.
  *
  * */



#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dummy/dummy_algorithm.h"
#include "cubeai/rl/algorithms/rl_algo_config.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/mountain_car_env.h"
#include "gymfcpp/time_step.h"

#include <iostream>


namespace example_0
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynMat;
using cubeai::DynVec;
using cubeai::rl::algos::DummyAlgorithm;
using cubeai::rl::algos::RLAlgoConfig;
using gymfcpp::MountainCar;

}

int main() {

    using namespace example_0;

    try{

        Py_Initialize();
        auto gym_module = boost::python::import("gym");
        auto gym_namespace = gym_module.attr("__dict__");

        MountainCar env("v0", gym_namespace, false);
        env.make();
        env.reset();

        auto config = RLAlgoConfig();
        config.n_episodes = 1000;
        config.n_itrs_per_episode = 100;
        config.render_environment = true;
        config.render_env_frequency = 10;

        DummyAlgorithm<MountainCar> agent(env, config);
        agent.train();
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


