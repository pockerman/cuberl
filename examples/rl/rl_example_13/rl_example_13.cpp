/**
 * This example illustrates how to use the REINFORCE algorithm
 * on the CartPole environment from Gymnasium
 **/
#include "cubeai/base/cubeai_config.h"

#if defined(USE_PYTORCH) && defined(USE_RLENVS_CPP)

#include "cubeai/base/cubeai_types.h"
#include "cubeai/io/csv_file_writer.h"
#include "cubeai/rl/algorithms/pg/simple_reinforce.h"
#include "cubeai/rl/trainers/rl_serial_agent_trainer.h"
#include "cubeai/maths/optimization/optimizer_type.h"
#include "cubeai/maths/optimization/pytorch_optimizer_factory.h"
#include "cubeai/maths/statistics/distributions/torch_categorical.h"

#include "rlenvs/envs/gymnasium/classic_control/cart_pole_env.h"
#include <torch/torch.h>

#include <unordered_map>
#include <iostream>
#include <string>
#include <any>
#include <filesystem>
#include <map>

namespace rl_example_13{


const std::string SERVER_URL = "http://0.0.0.0:8001/api";
const std::string EXPERIMENT_ID = "2";

namespace F = torch::nn::functional;

using cubeai::real_t;
using cubeai::uint_t;
using cubeai::torch_tensor_t;
using cubeai::rl::algos::pg::ReinforceSolver;
using cubeai::rl::algos::pg::ReinforceConfig;
using cubeai::rl::RLSerialAgentTrainer;
using cubeai::rl::RLSerialTrainerConfig;
using cubeai::maths::stats::TorchCategorical;
using rlenvs_cpp::envs::gymnasium::CartPole;


const uint_t L1 = 4;
const uint_t L2 = 150;
const uint_t L3 = 3;
const real_t LEARNING_RATE = 0.0009;


// The class that models the Policy network to train
class PolicyNetImpl: public torch::nn::Module
{
public:


    PolicyNetImpl();

    torch_tensor_t forward(torch_tensor_t state);

    template<typename StateTp>
    std::tuple<uint_t, real_t> act(const StateTp& state);
	
private:

   torch::nn::Linear fc1_;
   torch::nn::Linear fc2_;

};


PolicyNetImpl::PolicyNetImpl()
    :
      fc1_(torch::nn::Linear(4, 16)),
      fc2_(torch::nn::Linear(16, 2))
{
    register_module("fc1", fc1_);
    register_module("fc2", fc2_);
}

/*template<typename LossValuesTp>
void
PolicyNetImpl::update_policy_loss(const LossValuesTp& vals){

     torch_tensor_t torch_vals = torch::tensor(vals);

     // specify that we require the gradient
     loss_ = torch::cat(torch::tensor(vals, torch::requires_grad())).sum();
}

void
PolicyNetImpl::step_backward_policy_loss(){
    loss_.backward();
}*/

torch_tensor_t
PolicyNetImpl::forward(torch_tensor_t x){

    x = F::relu(fc1_->forward(x));
    x = fc2_->forward(x);
    return F::softmax(x, F::SoftmaxFuncOptions(0));
}


template<typename StateTp>
std::tuple<uint_t, real_t>
PolicyNetImpl::act(const StateTp& state){

    auto torch_state = torch::tensor(state);

    auto probs = forward(torch_state);
    auto m = TorchCategorical(probs, false);
    auto action = m.sample();
    return std::make_tuple(action.item().toLong(), 
	                       m.log_prob(action).item().to<real_t>());

}

TORCH_MODULE(PolicyNet);


struct Loss_1
{
	torch_tensor_t operator()(torch_tensor_t preds, torch_tensor_t y)const;
};

torch_tensor_t 
Loss_1::operator()(torch_tensor_t preds, torch_tensor_t y)const{
	return -1.0 * torch::sum(y * torch::log(preds));
}

typedef Loss_1 loss_type;
typedef CartPole env_type;
typedef PolicyNet policy_type;
typedef ReinforceSolver<env_type, PolicyNet, loss_type> solver_type;
}


int main(){

    using namespace rl_example_13;

    try{

        // let's create a directory where we want to
        //store all the results from running a simualtion
        std::filesystem::create_directories("experiments/" + EXPERIMENT_ID);
        torch::manual_seed(42);

        auto env = CartPole(SERVER_URL);
        std::cout<<"Environment URL: "<<env.get_url()<<std::endl;

        std::cout<<"Creating the environment..."<<std::endl;
        std::unordered_map<std::string, std::any> options;

        // with Gymnasium v0 is not working
        env.make("v1", options);
        env.reset();

        std::cout<<"Done..."<<std::endl;
        std::cout<<"Number of actions="<<env.n_actions()<<std::endl;

        PolicyNet policy;

        //auto optimizer_ptr = std::make_unique<torch::optim::Adam>(policy->parameters(),
        //                                                          torch::optim::AdamOptions(1e-2));

        // reinforce options
        ReinforceConfig opts = {true, 1000, 100, 100, 100, 1.0e-2, 0.1, 195.0};

        std::map<std::string, std::any> opt_options;
        opt_options.insert(std::make_pair("lr", LEARNING_RATE));

        auto pytorch_ops = cubeai::maths::optim::pytorch::build_pytorch_optimizer_options(cubeai::maths::optim::OptimzerType::ADAM,
                                                                                          opt_options);

        auto policy_optimizer = cubeai::maths::optim::pytorch::build_pytorch_optimizer(cubeai::maths::optim::OptimzerType::ADAM,
                                                                                       *policy, pytorch_ops);

		loss_type loss;
        solver_type solver(opts, policy, loss, policy_optimizer);


        RLSerialTrainerConfig config;
        config.n_episodes = 10;
        config.output_msg_frequency = 10;
        RLSerialAgentTrainer<env_type, solver_type> trainer(config, solver);
        trainer.train(env);

        auto info = trainer.train(env);
        std::cout<<"Trainer info: "<<info<<std::endl;

        // save the rewards per episode for visualization
        // purposes
        auto filename = std::string("experiments/") + EXPERIMENT_ID;
        filename += "/reinforce_rewards.csv";
        cubeai::io::CSVWriter csv_writer(filename, cubeai::io::CSVWriter::default_delimiter());
        csv_writer.open();
        csv_writer.write_column_vector(trainer.episodes_total_rewards());


        // save the policy also so that we can load it and check
        // use it
        auto policy_model_filename = std::string("experiments/") + 
		                             EXPERIMENT_ID + std::string("/reinforce_cartpole_policy.pth");
        //torch::save(policy,policy_model_filename);
        //auto model_scripted = torch::jit::scr //script(policy);
        //model_scripted.save(policy_model_filename);
        torch::serialize::OutputArchive archive;
        policy->save(archive);
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
