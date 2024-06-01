#ifndef TORCH_CATEGORICAL_H
#define TORCH_CATEGORICAL_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
#include "cubeai/maths/statistics/distributions/torch_distribution.h"

#include <torch/torch.h>


namespace cubeai {
namespace maths {
namespace stats {

class TorchCategorical final : public TorchDistribution
{


public:

    ///
    /// \brief TorchCategorical
    /// \param probs
    /// \param logits
    ///
    TorchCategorical(const torch_tensor_t *probs, const torch_tensor_t *logits);
    
    
    ///
    /// \brief ~TorchCategorical. Destructor
    ///
    virtual ~TorchCategorical();

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
    virtual torch_tensor_t sample(c10::ArrayRef<int64_t> sample_shape = {});

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
#endif
#endif
