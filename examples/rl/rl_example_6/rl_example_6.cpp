#include "cubeai/base/cubeai_config.h"

#ifdef USE_GYMFCPP

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/dp/iterative_policy_evaluation.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/rl/policies/uniform_discrete_policy.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/frozen_lake_env.h"
#include "gymfcpp/time_step.h"

#include <boost/python.hpp>

#include <iostream>



namespace rl_example_6
{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::policies::UniformDiscretePolicy;
using cubeai::rl::algos::dp::IterativePolicyEval;
using cubeai::rl::algos::dp::IterativePolicyEvalConfig;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
//typedef gymfcpp::TimeStep<uint_t> time_step_t;

typedef rlenvs_cpp::gymfcpp::FrozenLake<4> env_type;

}

int main() {

    using namespace rl_example_6;

    Py_Initialize();
    auto gym_module = boost::python::import("__main__");
    auto gym_namespace = gym_module.attr("__dict__");

    env_type env("v0", gym_namespace);
    env.make();

    UniformDiscretePolicy policy(env.n_states(), env.n_actions());
    IterativePolicyEvalConfig config;
    config.tolerance = 1.0e-8;

    IterativePolicyEval<env_type, UniformDiscretePolicy> algorithm(config, policy);

    RLSerialTrainerConfig trainer_config = {10, 10000, 1.0e-8};

    RLSerialAgentTrainer<env_type, IterativePolicyEval<env_type,
            UniformDiscretePolicy>> trainer(trainer_config, algorithm);

    auto info = trainer.train(env);
    std::cout<<info<<std::endl;

    // save the value function for plotting
    algorithm.save("iterative_policy_evaluation_frozen_lake.csv");

   return 0;
}
#else
#include <iostream>

int main() {

    std::cout<<"This example requires  gymfcpp. Configure cubeai to use gymfcpp"<<std::endl;
    return 0;
}
#endif


