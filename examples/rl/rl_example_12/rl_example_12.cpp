/**
 * This example illustrates how to use the REINFORCE algorithm
 * on the CartPole environment from Gymnasium
 *
 *
 *
 *
 * */
#include "cubeai/base/cubeai_config.h"

#if defined(USE_PYTORCH) && defined(USE_RLENVS_CPP)

#include "cubeai/base/cubeai_types.h"
#include "cubeai/io/csv_file_writer.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/maths/optimization/optimizer_type.h"
#include "cubeai/maths/optimization/pytorch_optimizer_factory.h"
#include "cubeai/rl/policies/epsilon_greedy_policy.h"

#include "rlenvs/envs/grid_world/grid_world_env.h"
#include <boost/log/trivial.hpp>
#include <torch/torch.h>

#include <unordered_map>
#include <iostream>
#include <string>
#include <any>
#include <filesystem>
#include <map>

namespace rl_example_12{


const std::string EXPERIMENT_ID = "1";

namespace F = torch::nn::functional;

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::torch_tensor_t;

using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using cubeai::rl::policies::EpsilonGreedyPolicy;
using rlenvs_cpp::envs::grid_world::Gridworld;

const uint_t l1 = 64;
const uint_t l2 = 150;
const uint_t l3 = 100;
const uint_t l4 = 4;
const uint_t SEED = 42;
const uint_t TOTAL_EPOCHS = 1000;
const uint_t TOTAL_ITRS_PER_EPOCH = 10;
const real_t GAMMA = 0.9;
const real_t EPSILON = 1.0;
const real_t LEARNING_RATE = 1.0e-3;


// The class that models the Policy network to train
class QNetImpl: public torch::nn::Module
{
public:


    QNetImpl();

    torch_tensor_t forward(torch_tensor_t);

    template<typename LossValuesTp>
    void update_policy_loss(const LossValuesTp& vals);

    void step_backward_policy_loss();

    torch_tensor_t compute_loss(){return loss_;}

private:

   torch::nn::Linear fc1_;
   torch::nn::Linear fc2_;
   torch::nn::Linear fc3_;

   // placeholder for the loss
   torch_tensor_t loss_;

};


QNetImpl::QNetImpl()
    :
      fc1_(torch::nn::Linear(l1, l2)),
      fc2_(torch::nn::Linear(l2, l3)),
      fc3_(torch::nn::Linear(l3, l4))
{
    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
    register_module("fc3", fc3_);
}

template<typename LossValuesTp>
void
QNetImpl::update_policy_loss(const LossValuesTp& vals){

     torch_tensor_t torch_vals = torch::tensor(vals);

     // specify that we require the gradient
     loss_ = torch::cat(torch::tensor(vals, torch::requires_grad())).sum();
}

void
QNetImpl::step_backward_policy_loss(){
    loss_.backward();
}

torch_tensor_t
QNetImpl::forward(torch_tensor_t x){

    x = F::relu(fc1_->forward(x));
    x = fc2_->forward(x);
    x = F::relu(fc3_->forward(x));
    return x; //F::softmax(x, F::SoftmaxFuncOptions(0));
}




TORCH_MODULE(QNet);

// 4x4 grid
typedef Gridworld<4> env_type;
typedef env_type::state_type state_type;


torch_tensor_t
state_to_torch_tensor(const state_type& state){


}


}


int main(){

    using namespace rl_example_12;

    try{

        BOOST_LOG_TRIVIAL(info)<<"Starting agent training...";
        BOOST_LOG_TRIVIAL(info)<<"Numebr of episodes to trina: "<<TOTAL_EPOCHS;

        // let's create a directory where we want to
        //store all the results from running a simualtion
        std::filesystem::create_directories("experiments/" + EXPERIMENT_ID);

        // set the seed for PyTorch
        torch::manual_seed(SEED);

        BOOST_LOG_TRIVIAL(info)<<"Creating the environment...";

        // create a 4x4 grid
        auto env = env_type();

        std::unordered_map<std::string, std::any> options;
        env.make("v1", options);

        BOOST_LOG_TRIVIAL(info)<<"Done...";
        BOOST_LOG_TRIVIAL(info)<<"Environment name: "<<env.name;
        BOOST_LOG_TRIVIAL(info)<<"Number of actions available: "<<env.n_actions();
        BOOST_LOG_TRIVIAL(info)<<"Number of states available: "<<env.n_states();


        // the network to train for the q values
        QNet qnet;

        auto optimizer_ptr = std::make_unique<torch::optim::Adam>(qnet->parameters(),
                                                                  torch::optim::AdamOptions(LEARNING_RATE));

        // we will use an epsilon-greedy policy
        EpsilonGreedyPolicy policy(EPSILON, SEED);

        // the loss function to use
        auto loss_fn = torch::nn::MSELoss();

        std::vector<real_t> losses;
        losses.reserve(TOTAL_ITRS_PER_EPOCH* TOTAL_EPOCHS);

        // loop over the epochs
        for(uint_t epoch=0; epoch < TOTAL_EPOCHS; ++epoch){

            BOOST_LOG_TRIVIAL(info)<<"Starting epoch: "<<epoch<<std::endl;

            // for every new epoch we reset the environment
            auto time_step = env.reset();

            auto done = false;

            while(!done){

                auto torch_state = state_to_torch_tensor(time_step.observation());

                // get the qvals
                auto qvals = qnet(torch_state);
                auto action_idx = policy(qvals);

                time_step = env.step(action_idx);

                if(time_step.done()){
                    break;
                }

                //
                torch_state = state_to_torch_tensor(time_step.observation());

                // tell the model that we don't use grad here
                qnet->eval();
                auto new_q_vals = qnet(torch_state);

                // find the maximum
                auto max_q = torch::max(new_q_vals);
                auto reward = time_step.reward();
                auto y = torch::tensor({reward});
                if(reward == -1.0){
                    y +=  max_q * GAMMA;
                }

                // create torch tensors
                //auto Y = torch_tensor_t([y]);
                auto X = qvals.squeeze()[action_idx];

                // calculate the loss
                auto loss = loss_fn(X, y); //torch::nn::functional::mse_loss(output, y_train);
                optimizer_ptr -> zero_grad();
                optimizer_ptr -> step();

                BOOST_LOG_TRIVIAL(info)<<"Loss at epoch: "<<loss.item<double>();
                losses.push_back(loss.item<double>());

            }

            BOOST_LOG_TRIVIAL(info)<<"Epoch finished..."<<std::endl;
        }

        //solver_type solver(opts, policy, policy_optimizer);


        //RLSerialTrainerConfig config;
        //config.n_episodes = 10;
        //config.output_msg_frequency = 10;

        //RLSerialAgentTrainer<env_type, solver_type> trainer(config, solver);
        //trainer.train(env);

        //auto info = trainer.train(env);
        //std::cout<<"Trainer info: "<<info<<std::endl;

        // save the rewards per episode for visualization
        // purposes
        auto filename = std::string("experiments/") + EXPERIMENT_ID;
        filename += "/reinforce_rewards.csv";
        cubeai::io::CSVWriter csv_writer(filename, cubeai::io::CSVWriter::default_delimiter());
        csv_writer.open();
        //csv_writer.write_column_vector(trainer.episodes_total_rewards());


        // save the policy also so that we can load it and check
        // use it
        auto policy_model_filename = std::string("experiments/") + EXPERIMENT_ID + std::string("/dqn_grid_world_policy.pth");
        //torch::save(policy,policy_model_filename);
        //auto model_scripted = torch::jit::scr //script(policy);
        //model_scripted.save(policy_model_filename);
        torch::serialize::OutputArchive archive;
        qnet -> save(archive);
        archive.save_to(policy_model_filename);

    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){
        std::cout<<"Unknown exception occured"<<std::endl;
    }
    return 0;
}
#else
#include <iostream>
int main(){

    std::cout<<"This example requires PyTorch and gymfcpp. Reconfigure cuberl with USE_PYTORCH and USE_RLENVS_CPP flags turned ON."<<std::endl;
    return 0;
}
#endif
