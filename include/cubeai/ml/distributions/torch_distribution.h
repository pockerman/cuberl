#ifndef TORCH_DISTRIBUTION_H
#define TORCH_DISTRIBUTION_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include "cubeai/base/cubeai_types.h"
#include "torch/torch.h"

#include <vector>

namespace cubeai {
namespace ml {
namespace stats {

///
/// \brief The TorchDistribution class. Base class to derive
/// distributions for PyTorch.
///
class TorchDistribution
{

public:

    ///
    /// \brief Destructor
    ///
    virtual ~TorchDistribution() = default;

    ///
    /// \brief entropy
    ///
    virtual torch_tensor_t entropy() = 0;

    ///
    /// \brief log_prob
    /// \param value
    /// \return
    ///
    virtual torch_tensor_t log_prob(torch_tensor_t value) = 0;

    ///
    /// \brief Ssample. Sample from the distribution
    ///
    virtual torch_tensor_t sample(c10::ArrayRef<int64_t> sample_shape) = 0;

protected:

    ///
    /// \brief batch_shape_
    ///
    std::vector<int64_t> batch_shape_;

    ///
    /// \brief event_shape_
    ///
    std::vector<int64_t> event_shape_;

    ///
    /// \brief extended_shape
    /// \param sample_shape
    /// \return
    ///
    std::vector<int64_t> extended_shape(c10::ArrayRef<int64_t> sample_shape);

};

}
}
}
#endif
#endif // TORCH_DISTRIBUTION_H
