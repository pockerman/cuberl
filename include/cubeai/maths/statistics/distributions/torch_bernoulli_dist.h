// SPDX-FileCopyrightText: 2024 <copyright holder> <email>
// SPDX-License-Identifier: Apache-2.0

/**
 * Major implementation is taken from  https://github.com/Omegastick/pytorch-cpp-rl/tree/master
 *
 */

#ifndef TORCH_BERNOULLI_DIST_H
#define TORCH_BERNOULLI_DIST_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
#include "cubeai/maths/statistics/distributions/torch_distribution.h"

#include <torch/torch.h>

namespace cuberl {
namespace maths {
namespace stats {

/**
 * @brief class  TorchBernoulliDist
 */
class TorchBernoulliDist final : public TorchDistributionBase
{
public:

     /**
     * @brief Default constructor
     *
     */
    TorchBernoulliDist() = default;

    /**
     * @brief Build the categorical distribution from given
     * torch_tensor_t of probabilities. if the flag
     * build_from_logits, then the given tensor is assumed
     * to represent logits
     *
     */
    TorchBernoulliDist(torch_tensor_t probs, bool do_build_from_logits=false);

    ///
    /// \brief ~TorchCategorical. Destructor
    ///
    virtual ~TorchBernoulliDist() = default;

    ///
    /// \brief entropy
    /// \return
    ///
    virtual torch_tensor_t entropy() override;

    ///
    /// \brief log_prob
    /// \param value
    /// \return
    ///
    virtual torch_tensor_t log_prob(torch_tensor_t value) override;

    ///
    ///
    ///
    virtual torch_tensor_t sample(c10::ArrayRef<int64_t> sample_shape = {})override;


    /**
     * @brief build the distribution form logits
     */
    void build_from_logits(torch_tensor_t logits);

    /**
     * @brief build the distribution from probabilities
     */
    void build_from_probabilities(torch_tensor_t probs);

    ///
    /// \brief get_logits
    ///
    torch_tensor_t get_logits()const { return logits_; }

    ///
    /// \brief get_probs
    /// \return
    ///
    torch_tensor_t get_probs()const { return probs_; }

private:
  torch_tensor_t probs_;
  torch_tensor_t logits_;
  torch_tensor_t param_;
  int num_events_;


};

}
}
}

#endif // TORCH_BERNOULLI_DIST_H
#endif
