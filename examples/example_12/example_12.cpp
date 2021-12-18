#include "cubeai/base/cubeai_config.h"

#if defined(USE_PYTORCH) && defined(USE_GYMFCPP)

#include "cubeai/base/cubeai_types.h"
#include "cubeai/rl/utilities/experience_buffer.h"
#include "cubeai/base/torch_tensor_utils.h"

#include "gymfcpp/gymfcpp_types.h"
#include "gymfcpp/cart_pole.h"
#include "gymfcpp/time_step.h"

#include <torch/torch.h>

#include <tuple>
#include <deque>
#include <iostream>

namespace F = torch::nn::functional;

namespace example{

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::torch_tensor_t;
using cubeai::torch_int_t;
using cubeai::rl::ExperienceBuffer;


// constants
const uint_t BATCH_SIZE = 128;
const real_t GAMMA = 0.999;
const real_t EPS_START = 0.9;
const real_t EPS_END = 0.05;
const real_t EPS_DECAY = 200;
const uint_t TARGET_UPDATE = 10;

const auto POLICY_NET_FILE = "policy_net.json";
const auto TARGET_NET_FILE = "target_net.json";


typedef gymfcpp::CartPole::time_step_t time_step_t;
typedef gymfcpp::CartPole::Screen screen_t;

// batch type to be used below
template<typename StateTp, typename ActionTp, typename RewardTp>
struct BatchType
{
    typedef screen_t state_t;
    typedef ActionTp action_t;
    typedef RewardTp reward_t;
    std::vector<std::tuple<state_t, action_t, reward_t, state_t, bool>> batch;

    void add_experience(const state_t& state, const action_t& action,  const reward_t& reward,
                        const state_t& next_state, bool done);

    // extract the non final next states from the batch
    std::vector<torch_tensor_t> get_non_final_next_states_as_torch_tensor();

    std::vector<torch_tensor_t> get_states_as_torch_tensor();

    torch_tensor_t get_actions_as_torch_tensor();

    torch_tensor_t get_rewards_as_torch_tensor();

    torch_tensor_t get_non_final_next_states_mask();

};

template<typename StateTp, typename ActionTp, typename RewardTp>
void
BatchType<StateTp, ActionTp, RewardTp>::add_experience(const state_t& state, const action_t& action,  const reward_t& reward,
                                                       const state_t& next_state, bool done){

    batch.push_back(std::make_tuple(state, action, reward, next_state, done));
}

template<typename StateTp, typename ActionTp, typename RewardTp>
std::vector<torch_tensor_t>
BatchType<StateTp, ActionTp, RewardTp>::get_non_final_next_states_as_torch_tensor(){

    std::vector<torch_tensor_t> next_states;

    for(const auto& item : batch){

        if(std::get<4>(item)){
            next_states.push_back(std::get<3>(item).get_as_torch_tensor());
        }
    }

    return next_states;
}


template<typename StateTp, typename ActionTp, typename RewardTp>
torch_tensor_t
BatchType<StateTp, ActionTp, RewardTp>::get_non_final_next_states_mask(){

    std::vector<int> next_states;

    for(const auto& item : batch){

        if(std::get<4>(item)){
            next_states.push_back(0);
        }
        else{

            next_states.push_back(1);
        }
    }

    return cubeai::torch_utils::create_mask(next_states);
}

template<typename StateTp, typename ActionTp, typename RewardTp>
std::vector<torch_tensor_t>
BatchType<StateTp, ActionTp, RewardTp>::get_states_as_torch_tensor(){

    std::vector<torch_tensor_t> states;

    for(const auto& item : batch){
        states.push_back(std::get<0>(item).get_as_torch_tensor());
    }

    return states;
}

template<typename StateTp, typename ActionTp, typename RewardTp>
torch_tensor_t
BatchType<StateTp, ActionTp, RewardTp>::get_actions_as_torch_tensor(){

    std::vector<torch_int_t> actions;

    for(const auto& item : batch){
       actions.push_back(std::get<1>(item));
    }

    return torch::tensor(actions);

}

template<typename StateTp, typename ActionTp, typename RewardTp>
torch_tensor_t
BatchType<StateTp, ActionTp, RewardTp>::get_rewards_as_torch_tensor(){


    std::vector<real_t> rewards;

    for(const auto& item : batch){
       rewards.push_back(std::get<2>(item));
    }

    return torch::tensor(rewards);

}

struct ExperienceType
{
    torch_tensor_t state;
    uint_t action;
    torch_tensor_t next_state;
    real_t reward;
    bool done;

};


typedef ExperienceType experience_t;


// Environment wrapper
class Environment
{
public:


    explicit Environment(gymfcpp::obj_t gym_namespace);
    void build();
    uint_t n_actions()const{return env_.n_actions();}

    // get the current screen
    gymfcpp::CartPole::Screen get_screen()const{return env_.get_screen();}

    template<typename ActionTp>
    auto step(const ActionTp& a){return env_.step(a);}

    auto reset(){return env_.reset();}


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

    CartPoleDQNAgent(torch::Device device, CartPoleDQN policy_net,
                     CartPoleDQN target_net, const std::string& policy_net_path,
                     const std::string& target_net_path);

    void train_step();
    void train();

    template<typename ActionTp>
    gymfcpp::TimeStep<time_step_t> step(const ActionTp& a){return env_ptr_->step(a);}

    template<typename StateTp>
    uint_t select_action(const StateTp& state);

private:

    // the device to use
    torch::Device device_;

    // the target and policy networks
    CartPoleDQN policy_net_;
    CartPoleDQN target_net_;

    // the optimizer
    torch::optim::RMSprop optimizer_;

    // memory buffer
    ExperienceBuffer<experience_t> memory_;
    uint_t num_episodes_;

    Environment* env_ptr_;

    std::string policy_net_path_;
    std::string target_net_path_;

};

CartPoleDQNAgent::CartPoleDQNAgent(torch::Device device, CartPoleDQN policy_net,
                                   CartPoleDQN target_net, const std::string& policy_net_path,
                                   const std::string& target_net_path)
    :
    device_(device),
    policy_net_(policy_net),
    target_net_(target_net),
    optimizer_(policy_net_->parameters()),
    memory_(10000),
    env_ptr_(nullptr),
    policy_net_path_(policy_net_path),
    target_net_path_(target_net_path)
{}


template<typename StateTp>
uint_t
CartPoleDQNAgent::select_action(const StateTp& state){
    return 1;
}

void
CartPoleDQNAgent::train_step(){

    if(memory_.size() < BATCH_SIZE)
        return;

    BatchType<screen_t, uint_t, real_t>  batch;
    memory_.sample(BATCH_SIZE, batch);

    // get the non final next states
    auto non_final_next_states = batch.get_non_final_next_states_as_torch_tensor();

    auto torch_non_final_next_states = torch::cat(non_final_next_states);

    // get the batches
    // Compute a mask of non-final states and concatenate the batch elements
    // (a final state would've been the one after which simulation ended)
    //auto non_final_next_states = torch::cat([s for s in batch.next_state if s is not None])

    auto state_batch =  torch::cat(batch.get_states_as_torch_tensor());
    auto action_batch = torch::cat(batch.get_actions_as_torch_tensor());
    auto reward_batch = torch::cat(batch.get_rewards_as_torch_tensor());
    auto non_final_mask = batch.get_non_final_next_states_mask();

    // Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    // columns of actions taken. These are the actions which would've been taken
    // for each batch state according to policy_net
    auto state_action_values = policy_net_->forward(state_batch).gather(1, action_batch);

    // Compute V(s_{t+1}) for all next states.
    // Expected values of actions for non_final_next_states are computed based
    // on the "older" target_net; selecting their best reward with max(1)[0].
    // This is merged based on the mask, such that we'll have either the expected
    // state value or 0 in case the state was final.
    auto next_state_values = torch::zeros(BATCH_SIZE, device_);
    auto target_values = std::get<0>(target_net_->forward(torch_non_final_next_states).max(1)).detach();

    // fill in the next state values according to the mask
    next_state_values.index_put_({non_final_mask}, target_values);

    // Compute the expected Q values
    auto expected_state_action_values = (next_state_values * GAMMA) + reward_batch;

    // Compute Huber loss
    auto criterion = torch::nn::SmoothL1Loss();
    auto loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1));

    // Optimize the model
    optimizer_.zero_grad();
    loss.backward();
    for(auto& param : policy_net_->parameters())
        param.grad().data().clamp_(-1, 1);
    optimizer_.step();
}

void
CartPoleDQNAgent::train(){

    for(uint_t e=0; e<num_episodes_; ++e){

        env_ptr_->reset();

        // these should be torch tensors
        auto last_screen = env_ptr_->get_screen().get_as_torch_tensor();
        auto current_screen = env_ptr_->get_screen().get_as_torch_tensor();

        auto state = current_screen - last_screen;

        for(uint_t itr=0; itr < 1000; ++itr){

            // select an action
            auto action = select_action(state);

            // step in the environment
            auto step_time = env_ptr_->step(action);

            // update states
            last_screen = current_screen;
            current_screen = env_ptr_->get_screen().get_as_torch_tensor();

            if(!step_time.done()){
                auto next_state = current_screen - last_screen;
                experience_t experience = {current_screen, action, last_screen, step_time.reward(), step_time.done()};
                memory_.append(experience);
                state = next_state;
            }
            else{

               experience_t experience = {current_screen, action, last_screen, step_time.reward(), step_time.done()};
               memory_.append(experience);
            }

            // optimize the model
            train_step();

            if(step_time.done()){
                break;
            }
        }

        // update the target
        if(e % TARGET_UPDATE == 0){
            torch::save(policy_net_, POLICY_NET_FILE);
            torch::load(target_net_, POLICY_NET_FILE);
        }
    }

}

}


int main(){

    using namespace example;

    try{

        std::cout<<"INFO: Starting exaple..."<<std::endl;

        torch::Device device = torch::kCPU;
        if (torch::cuda::is_available()) {
          std::cout << "CUDA is available! Training on GPU." << std::endl;
          device = torch::kCUDA;
        }

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
        torch::save(policy_net, POLICY_NET_FILE);

        auto target_net = CartPoleDQN(screen_height, screen_width, n_actions);

        torch::load(target_net, POLICY_NET_FILE);
        target_net->eval();

        CartPoleDQNAgent agent(device, policy_net, target_net, POLICY_NET_FILE, TARGET_NET_FILE);
        agent.train();


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

    std::cout<<"This example requires PyTorch and gymfccp. Configure cubeai with PyTorch and gymfcpp support"<<std::endl;
    return 0;
}
#endif
