#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/utilities/experience_buffer.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/cart_pole.h"
#include "gymfcpp/time_step.h"

#include <torch/torch.h>

#include <deque>
#include <iostream>

namespace F = torch::nn::functional;

namespace example{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::rl::ExperienceBuffer;


// constants
const uint_t BATCH_SIZE = 128;
const real_t GAMMA = 0.999;
const real_t EPS_START = 0.9;
const real_t EPS_END = 0.05;
const real_t EPS_DECAY = 200;
const uint_t TARGET_UPDATE = 10;


class Screen;



// Environment wrapper
class Environment
{
public:


    explicit Environment(gymfcpp::obj_t gym_namespace);
    void build();
    uint_t n_actions()const{return env_.n_actions();}

    // get the current screen
    Screen get_screen();


private:

    //
    gymfcpp::CartPole env_;

};

Environment::Environment(gymfcpp::obj_t gym_namespace)
    :
      env_("v0", gym_namespace)
{}

void
Environment::build(){
    env_.make();
}


//
class CartPoleDQNImpl: public torch::nn::Module
{
public:

    //
    CartPoleDQNImpl(uint_t h, uint_t w, uint_t output);

    // forward
    torch::Tensor forward(torch::Tensor input);

private:


    torch::nn::Conv2d conv1_;
    torch::nn::BatchNorm2d bn1_;
    torch::nn::Conv2d conv2_;
    torch::nn::BatchNorm2d bn2_;
    torch::nn::Conv2d conv3_;
    torch::nn::BatchNorm2d bn3_;
    torch::nn::Linear linear_;

    // helper function
    uint_t conv2d_size_out_(uint_t size, uint_t kernel_size=5, uint_t stride=2)const{return (size - (kernel_size - 1) - 1) / stride +1;}
    uint_t get_linear_input_size_(uint_t h, uint_t w) const;

};


CartPoleDQNImpl::CartPoleDQNImpl(uint_t h, uint_t w, uint_t output_size)
    :
     torch::nn::Module(),
     conv1_(torch::nn::Conv2dOptions(3, 16, 5).stride(2)),
     bn1_(16),
     conv2_(torch::nn::Conv2dOptions(16, 32, 5).stride(2)),
     bn2_(32),
     conv3_(torch::nn::Conv2dOptions(32, 32, 5).stride(2)),
     bn3_(32),
     linear_(torch::nn::Linear(get_linear_input_size_(h, w), output_size))


{

    // register_module() is needed if we want to use the parameters() method later on
    register_module("linear", linear_);
    register_module("conv1", conv1_);
    register_module("conv2", conv2_);
    register_module("conv3", conv3_);
    register_module("bn1", bn1_);
    register_module("bn2", bn2_);
    register_module("bn3", bn3_);

}

uint_t
CartPoleDQNImpl::get_linear_input_size_(uint_t h, uint_t w) const{

    auto convw = conv2d_size_out_(conv2d_size_out_(conv2d_size_out_(w)));
    auto convh = conv2d_size_out_(conv2d_size_out_(conv2d_size_out_(h)));
    auto linear_input_size = convw * convh * 32;
    return linear_input_size;
}

torch::Tensor
CartPoleDQNImpl::forward(torch::Tensor x){

    x = F::relu(bn1_->forward(conv1_->forward(x)));
    x = F::relu(bn2_->forward(conv2_->forward(x)));
    x = F::relu(bn3_->forward(conv3_->forward(x)));
    return linear_->forward(x.view({x.size(0), -1}));

}

TORCH_MODULE(CartPoleDQN);

class InvalidState;

// the class that wraps the agent
class CartPoleDQNAgent
{

public:

    void train_step();
    void train();

    template<typename ActionTp>
    gymcpp::TimeStep step(const ActionTp& a){return env_ptr_->step(a);}

    template<typename StateTp>
    uint_t select_action(const StateTp& state);

private:

    CartPoleDQN policy_net_;
    CartPoleDQN target_net_;

    torch::optim::RMSprop optimizer_;
    ExperienceBuffer<gymcpp::TimeStep> memory_;
    uint_t num_episodes_;

    Environment* env_ptr_;

};

void
CartPoleDQNAgent::train_step(){

    if(memory_.size() < BATCH_SIZE)
        return;

    auto transitions = memory_.sample(BATCH_SIZE);

    // get the bathes

    // Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    // columns of actions taken. These are the actions which would've been taken
    // for each batch state according to policy_net
    auto state_action_values = policy_net_.operator()(state_batch).gather(1, action_batch);

    // Compute V(s_{t+1}) for all next states.
    // Expected values of actions for non_final_next_states are computed based
    // on the "older" target_net; selecting their best reward with max(1)[0].
    // This is merged based on the mask, such that we'll have either the expected
    // state value or 0 in case the state was final.
    auto next_state_values = torch.zeros(BATCH_SIZE, device=device);
    next_state_values[non_final_mask] = target_net_->operator()(non_final_next_states).max(1)[0].detach();

    // Compute the expected Q values
    auto expected_state_action_values = (next_state_values * GAMMA) + reward_batch;

    // Compute Huber loss
    auto criterion = torch::nn::SmoothL1Loss();
    auto loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1));

    // Optimize the model
    optimizer_.zero_grad();
    loss.backward();
    for(auto& param : policy_net_->parameters())
        param.grad.data.clamp_(-1, 1);
    optimizer_.step();
}

void
CartPoleDQNAgent::train(){

    for(uint_t e=0; e<num_episodes_; ++e){

        env_ptr_->reset();

        auto last_screen = env_ptr_->get_screen();
        auto current_screen = env_ptr_->get_screen();

        auto state = current_screen - last_screen;

        for(uint_t itr=0; itr < 1000; ++itr){

            // select an action
            auto action = select_action(state);

            // step in the environment
            auto step_time = env_ptr_->step(action);

            // update states
            last_screen = current_screen;
            current_screen = env_ptr_->get_screen();

            i(!step_time.done()){
                auto next_state = current_screen - last_screen;
                memory_.add(state, action, next_state, step_time.reward());
                state = next_state;
            }
            else{
               memory_.add(state, action, next_state, step_time.reward());
               state = INVALID_STATE;
            }

            // optimize the model
            train_step();

            if(step_time.done()){
                break;
            }
        }

        // update the target
        if(e % TARGET_UPDATE == 0){
            target_net_->load(policy_net_->get_state());
        }
    }

}

}


int main(){

    using namespace example;

    try{

        std::cout<<"INFO: Starting exaple..."<<std::endl;

        Py_Initialize();

        auto gym_module = boost::python::import("gym");
        auto gym_namespace = gym_module.attr("__dict__");

        std::cout<<"INFO: Building environment..."<<std::endl;
        Environment env(gym_namespace);
        env.build();
        std::cout<<"Number of actions="<<env.n_actions()<<std::endl;

        auto screen_height = 5;
        auto screen_width = 10;
        auto n_actions = env.n_actions();

        auto policy_net = CartPoleDQN(screen_height, screen_width, n_actions);
        auto target_net = CartPoleDQN(screen_height, screen_width, n_actions);
        target_net->load_state_dict(policy_net.state_dict());
        target_net->eval();

        // the optimizer
        //auto optimizer = torch::optim::RMSprop(policy_net->parameters());
        //auto memory = ExperienceBuffer<gymcpp::TimeStep>(10000);


    }
    catch(std::exception& e){
        std::cout<<e.what()<<std::endl;
    }
    catch(...){

        std::cout<<"Unknown exception occured"<<std::endl;
    }
    return 0;
}
