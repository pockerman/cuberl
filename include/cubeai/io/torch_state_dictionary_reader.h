#ifndef TORCH_STATE_DICTIONARY_READER_H
#define TORCH_STATE_DICTIONARY_READER_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH

#include <torch/torch.h>
#include <nlohmann/json.hpp>

#include <string>
namespace cubeai {
namespace torch_utils {

///
/// \brief The TorchStateDictionaryReader class
///
class TorchStateDictionaryReader
{
public:

    ///
    /// \brief TorchStateDictionaryReader
    ///
    TorchStateDictionaryReader(const std::string& params_path, bool do_read=false);

    ///
    /// \brief read
    ///
    void read();

private:

    std::string params_path_;

    ///
    /// \brief j_ The json object that holds the
    /// read parameters.
    ///
    nlohmann::json j_;


};

}

}

#endif
#endif // TORCH_STATE_READER_H
