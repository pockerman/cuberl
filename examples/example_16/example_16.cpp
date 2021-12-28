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
#include "gymfcpp/gymfcpp_consts.h"

#include <boost/python.hpp>

#include <vector>
#include <iostream>
#include <unordered_map>
#include <utility>
#include <limits>
#include <map>


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

/**
 * Implementation of Monte Carlo tree serach for the OpenAI-Gym
 * environment.
 */
template<typename Env>
class TaxiMCTreeSearch: public MCTreeSearchBase<Env>
{
public:

    typedef typename Env::time_step_type time_step_type;
    typedef typename MCTreeSearchBase<Env>::env_type env_type;
    typedef typename MCTreeSearchBase<Env>::node_type node_type;

    //
    TaxiMCTreeSearch(MCTreeSearchConfig config, Env& env);


    time_step_type simulate_node(std::shared_ptr<node_type> node, env_type& env) override final;
    void on_episode() override final;
    void expand_node(std::shared_ptr<node_type> node, env_type& env) override final;
    void backprop(std::shared_ptr<node_type> node) override final;

private:

    // the reward the agent receives for every
    // training iteration within an episode
    real_t sum_reward_;

    //
    std::vector<typename Env::action_type> best_actions_;
    std::vector<real_t> best_rewards_;
    std::vector<typename Env::action_type> actions_;


};

template<typename Env>
TaxiMCTreeSearch<Env>::TaxiMCTreeSearch(MCTreeSearchConfig config, Env& env)
    :
    MCTreeSearchBase<Env>(config, env),
    sum_reward_(0.0),
    best_actions_(),
    best_rewards_()
{}


template<typename Env>
typename  TaxiMCTreeSearch<Env>::time_step_type
TaxiMCTreeSearch<Env>::simulate_node(std::shared_ptr<node_type> node, env_type& env){

    auto time_step = TaxiMCTreeSearch<Env>::time_step_type();
    while(node -> has_children()){

        if(node -> n_explored_children() < node -> n_children() ){

            auto child = node->get_child(node->n_explored_children());
            node ->update_explored_children();
            node = child;
        }
        else{

            node = node -> max_ucb_child(this->temperature_);

        }

        time_step  = env.step(node ->get_action());
        sum_reward_ += time_step.reward();
        actions_.push_back(node ->get_action());
    }

    return time_step;
}

template<typename Env>
void
TaxiMCTreeSearch<Env>::on_episode(){

    auto best_reward = std::numeric_limits<real_t>::min();

    std::map<std::string, std::string> names;

    names[gymfcpp::PY_ENV_NAME] = this->env_.py_env_name + "_" + std::to_string(this->current_episode_idx());
    names[gymfcpp::PY_RESET_ENV_RESULT_NAME] = this->env_.py_reset_result_name + "_" + std::to_string(this->current_episode_idx());
    names[gymfcpp::PY_STEP_ENV_RESULT_NAME] = this->env_.py_step_result_name + "_" + std::to_string(this->current_episode_idx());
    names[gymfcpp::PY_STATE_NAME] = this->env_.py_state_name + "_" + std::to_string(this->current_episode_idx());

    // simulate the environment
    auto env_copy = this->env_.copy(std::move(names));

    while(this->itr_mix_.continue_iterations()){

        sum_reward_ = 0.0;
        auto terminal = false;
        actions_.clear();


        auto node = this->root_;

        auto time_step = this->simulate_node(node, env_copy);

        terminal = time_step.done();

        if(!terminal){
            // expand the node if this is not
            // terminal
            this->expand_node(node, env_copy);
        }

        // creating exhaustive list of actions
        while(!terminal){
           auto action = env_copy.sample_action();
           auto new_time_step = env_copy.step(action);
           sum_reward_ += new_time_step.reward();
           actions_.push_back(action);

           if(actions_.size() > this->max_depth_tree()){
               sum_reward_ -= 100;
               break;
           }
        }

        // do some book keeping retaining the best reward value and actions
        if( best_reward < sum_reward_){
            best_reward = sum_reward_;
            best_actions_ = actions_;
        }

        // backpropagating in MCTS for assigning reward value to a node.
        this->backprop(node);
    }

    // step in the best actions
    sum_reward_ = 0.;
    for(auto action : best_actions_){
        auto time_step = this->env_.step(action);

         sum_reward_ += time_step.reward();
         if(time_step.done()){
             break;
         }
    }


    best_rewards_.push_back(sum_reward_);
}

template<typename Env>
void
TaxiMCTreeSearch<Env>::expand_node(std::shared_ptr<node_type> node, env_type& env){

    for(uint_t a=0; a < env.n_actions(); ++a ){
        node -> add_child(node, a);
    }

    node->shuffle_children();

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
        config.temperature = 1.0;
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
