#ifndef TORCH_AGENT_BASE_H
#define TORCH_AGENT_BASE_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
//#include "cubeai/rl/episode_result_info.h"
#include "cubeai/optimization/optimizer_type.h"

#include <torch/torch.h>
#include <boost/noncopyable.hpp>
#include <string>
#include <any>
#include <map>

namespace cubeai {
namespace rl{

/// Forward declaration
struct EpisodeResultInfo;

namespace pytorch{

///
/// \brief The TorchAgentConfig struct
///
struct TorchAgentConfig
{
    ///
    ///
    ///
    uint_t n_itrs_per_episode{0};

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

};

///
///
///
template<typename EnvType>
class TorchAgentBase: private boost::noncopyable
{
public:

    typedef EnvType env_type;
    typedef typename env_type::time_step_type time_step_type;

    ///
    ///
    ///
    virtual ~TorchAgentBase()=default;

    ///
    /// \brief actions_before_training_begins.  Execute any actions
    /// the algorithm needs before starting the episode
    ///
    virtual void actions_before_training_begins(env_type&)=0;

    ///
    /// \brief actions_before_episode_begins. Execute any actions the algorithm needs before
    /// starting the episode
    ///
    virtual void actions_before_episode_begins(env_type&, uint_t)=0;

    ///
    /// \brief on_training_episode
    /// \param env
    /// \param episode_idx
    /// \return
    ///
    virtual EpisodeResultInfo on_training_episode(env_type& env, uint_t episode_idx)=0;

    ///
    /// \brief  actions_after_episode_ends. Execute any actions the algorithm needs after
    /// ending the episode
    ///
    virtual void actions_after_episode_ends(env_type&, uint_t)=0;

    ///
    /// \brief actions_after_training_ends. Execute any actions the algorithm needs after
    /// the iterations are finished
    ///
    virtual void actions_after_training_ends(env_type&)=0;

protected:

    ///
    /// \brief TorchAgentBase
    /// \param config
    ///
    TorchAgentBase(torch::nn::Module& model);

    ///
    /// \brief model_ The torch model to train
    ///
    torch::nn::Module& model_;

};

template<typename EnvType>
TorchAgentBase<EnvType>::TorchAgentBase(torch::nn::Module& model)
    :
      model_(model)
{}

}
}
}
#endif
#endif // TORCH_AGENT_BASE_H
