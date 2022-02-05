#ifndef PYTORCH_SUPERVISED_TRAINER_H
#define PYTORCH_SUPERVISED_TRAINER_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_consts.h"
#include "cubeai/optimization/optimizer_type.h"
#include "cubeai/optimization/pytorch_optimizer_factory.h"
#include "cubeai/ml/pytorch_loss_wrapper.h"
#include <torch/torch.h>

#include <boost/noncopyable.hpp>
#include <iostream>
#include <map>
#include <any>

namespace cubeai{
namespace ml{
namespace pytorch{

///
/// \brief The PyTorchSupervisedTrainerConfig struct
///
struct PyTorchSupervisedTrainerConfig
{
    ///
    /// \brief n_epochs_
    ///
    uint_t n_epochs;

    ///
    /// \brief batch_size
    ///
    uint_t batch_size;

    ///
    ///
    ///
    bool show_loss_info{true};

    ///
    /// \brief device
    ///
    torch::Device device;

    ///
    /// \brief optim_type
    ///
    cubeai::optim::OptimzerType optim_type;

    ///
    /// \brief optim_options
    ///
    std::map<std::string, std::any> optim_options;

    ///
    ///
    ///
    PyTorchSupervisedTrainerConfig();
};

inline
PyTorchSupervisedTrainerConfig::PyTorchSupervisedTrainerConfig()
    :
    n_epochs(100),
    batch_size(50),
    show_loss_info(true),
    device(torch::kCPU),
    optim_type(cubeai::optim::OptimzerType::INVALID_TYPE),
    optim_options()
{}


template<typename ModelType>
class PyTorchSupervisedTrainer: private boost::noncopyable
{
public:

    typedef ModelType model_type;

    ///
    ///
    ///
    PyTorchSupervisedTrainer(PyTorchSupervisedTrainerConfig config, model_type& model);


    ///
    ///
    ///
    template<typename TrainDataLoaderType>
    void train(TrainDataLoaderType& data_loader, const PyTorchLossWrapper& wrapper );

protected:

    ///
    /// \brief config_
    ///
    PyTorchSupervisedTrainerConfig config_;

    ///
    /// \brief model_ptr_
    ///
    model_type& model_;

};

template<typename ModelType>
PyTorchSupervisedTrainer<ModelType>::PyTorchSupervisedTrainer(PyTorchSupervisedTrainerConfig config, model_type& model)
    :
    config_(config),
    model_(model)
{}

template<typename ModelType>
template<typename TrainDataSetType>
void
PyTorchSupervisedTrainer<ModelType>::train(TrainDataSetType& data_set, const PyTorchLossWrapper& loss_wrapper ){

    // create the optimizer and
    // Set floating point output precision
    std::cout << std::fixed << std::setprecision(4);

    // Data loaders
    auto train_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(
            std::move(data_set), config_.batch_size);

    // Number of samples in the training set
    const auto num_train_samples = data_set.size().value();
    const auto n_epochs = config_.n_epochs;

    const auto optim_ops = cubeai::optim::pytorch::build_pytorch_optimizer_options(config_.optim_type, config_.optim_options);

    // build optimizer
    auto optimizer = cubeai::optim::pytorch::build_pytorch_optimizer(config_.optim_type, model_, *optim_ops.get());


    // Train the model
    for (size_t epoch = 0; epoch != n_epochs; ++epoch) {

        // Initialize running metrics
        auto running_loss = 0.0;
        uint_t num_correct = 0;

        for (auto& batch : *train_loader) {

            auto data = batch.data.view({static_cast<long int>(config_.batch_size), -1}).to(config_.device);
            auto target = batch.target.to(config_.device);

            // Forward pass
            auto output = model_.forward(data);

            // Calculate loss. Use cross_entropy
            auto loss = loss_wrapper.calculate(output, target); //torch::nn::functional::cross_entropy(output, target);

            // Update running loss
            running_loss += loss. template item<real_t>() * data.size(0);

            // Calculate prediction
            auto prediction = output.argmax(1);

            // Update number of correctly classified samples
            num_correct += prediction.eq(target).sum(). template item<int64_t>();

            // Backward pass and optimize
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();
        }

        auto sample_mean_loss = running_loss / num_train_samples;
        auto accuracy = static_cast<real_t>(num_correct) / num_train_samples;

        std::cout<<cubeai::CubeAIConsts::info_str()<< "Epoch [" << (epoch + 1) << "/" << n_epochs << "], Trainset - Loss: "
            << sample_mean_loss << ", Accuracy: " << accuracy << '\n';
    }
}


}
}
}

#endif

#endif // PYTORCH_SUPERVISED_TRAINER_H
