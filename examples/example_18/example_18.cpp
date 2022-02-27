/**
  * Double Q-learning on CartPole-v0 environment
  *
 */


#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/math_constants.h"
#include "cubeai/rl/algorithms/td/double_q_learning.h"
#include "cubeai/rl/algorithms/td/td_algo_base.h"
#include "cubeai/rl/policies/epsilon_double_qtable_greedy_policy.h"
#include "cubeai/rl/epsilon_decay_options.h"
#include "gymfcpp/state_aggregation_cart_pole_env.h"

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


namespace example18{

typedef boost::python::api::object obj_t;
using cubeai::real_t;
using cubeai::uint_t;
using cubeai::DynVec;
using cubeai::rl::algos::td::DoubleQLearning;
using cubeai::rl::algos::td::TDAlgoConfig;
using cubeai::rl::policies::EpsilonDoubleQTableGreedyPolicy;
using cubeai::rl::EpsilonDecayOptionType;
using gymfcpp::StateAggregationCartPole;

const real_t EPS = 0.1;
const real_t GAMMA = 1.0;
const real_t ALPHA = 0.1;

// type of the table to use as Qtable
typedef typename std::map<std::tuple<uint_t, uint_t, uint_t, uint_t>, DynVec<real_t>> table_type;


}


int main(){

    using namespace example18;

    try{

        Py_Initialize();
        auto main_module = boost::python::import("__main__");
        auto main_namespace = main_module.attr("__dict__");

        // create the environment
        StateAggregationCartPole env("v0", main_namespace, 10);

        // the policy to use
        EpsilonDoubleQTableGreedyPolicy<table_type> policy(EPS, env.n_actions(), EpsilonDecayOptionType::NONE);

        // configuration for the algorithm
        TDAlgoConfig config;
        config.eta = ALPHA;
        config.gamma = GAMMA;
        config.n_episodes = 50000;
        config.n_itrs_per_episode = 10000;

        // the agent to traain
        DoubleQLearning<StateAggregationCartPole, EpsilonDoubleQTableGreedyPolicy<table_type>, table_type> agent(config, env, policy);
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
