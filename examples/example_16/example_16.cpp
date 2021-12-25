/**
  * Simple implementation of Monte Carlo tree serach
  * algorithm on the Taxi environment. The code below is a translation of
  * the Python code in
  * https://github.com/ashishrana160796/prototyping-self-driving-agents/blob/master/milestone-four/monte_carlo_tree_search_taxi_v3.ipynb
  *
 */


#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/algorithms/mc/mc_tree_search_base.h"
#include "cubeai/utils/array_utils.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/taxi.h"
#include "gymfcpp/time_step.h"

#include <boost/python.hpp>

#include <vector>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <limits>


namespace example{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::algos::MCTreeSearchBase;
using cubeai::rl::algos::MCTreeSearchConfig;

using gymfcpp::Taxi;

typedef Taxi::time_step_type time_step_type;


const real_t GAMMA = 1.0;
const uint_t N_EPISODES = 20000;
const uint_t N_ITRS_PER_EPISODE = 2000;
const real_t TOL = 1.0e-8;


template<typename Env>
class TaxiMCTreeSearch: public MCTreeSearchBase<Env>
{
public:

    typedef typename MCTreeSearchBase<Env>::node_type node_type;

    //
    TaxiMCTreeSearch(MCTreeSearchConfig config, Env& env);


    void on_episode();
    void expand_node(std::shared_ptr<node_type> node);
    void backprop(std::shared_ptr<node_type> node);

private:

    real_t sum_reward_;


};

template<typename Env>
TaxiMCTreeSearch<Env>::TaxiMCTreeSearch(MCTreeSearchConfig config, Env& env)
    :
    MCTreeSearchBase<Env>(config, env)
{}

template<typename Env>
void
TaxiMCTreeSearch<Env>::on_episode(){

    auto best_reward = std::numeric_limits<real_t>::min();

    while(this->itr_mix_.continue_iterations()){

    }

}

template<typename Env>
void
TaxiMCTreeSearch<Env>::expand_node(std::shared_ptr<node_type> node){

}

template<typename Env>
void
TaxiMCTreeSearch<Env>::backprop(std::shared_ptr<node_type> node){

    while(node){

        node -> update_visits();
        node -> update_total_score(sum_reward_);
        node = node ->parent();
    }

}

}


int main(){

    using namespace example;

    try{

        Py_Initialize();
        auto main_module = boost::python::import("__main__");
        auto main_namespace = main_module.attr("__dict__");

        Taxi env("v0", main_namespace, false);
        env.make();

        MCTreeSearchConfig config;
        config.max_tree_depth = 512;
        TaxiMCTreeSearch<Taxi> agent(config, env);
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
