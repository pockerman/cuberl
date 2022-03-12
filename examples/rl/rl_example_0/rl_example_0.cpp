/**
  * Example 0: Demonstrates the use of the DummyAlgorithm class.
  * This class exposes the basic API that most implemented RL
  * algorithms expose.
  *
  * */


#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dummy/dummy_algorithm.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
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
using cubeai::rl::algos::DummyAlgorithmConfig;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using gymfcpp::MountainCar;

}

int main() {

    using namespace example_0;

    try{

        Py_Initialize();
        auto main_module = boost::python::import("__main__");
        auto gym_namespace = main_module.attr("__dict__");

        // create the environment
        MountainCar env("v0", gym_namespace, false);
        env.make();
        env.reset();


        DummyAlgorithmConfig config = {100};
        DummyAlgorithm<MountainCar> algorithm(config);

        RLSerialTrainerConfig trainer_config = {100, 1000, 1.0e-8};

        RLSerialAgentTrainer<MountainCar, DummyAlgorithm<MountainCar>> trainer(trainer_config, algorithm);

        auto info = trainer.train(env);
        std::cout<<info<<std::endl;
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


