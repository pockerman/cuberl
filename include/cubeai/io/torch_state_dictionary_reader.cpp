#include "cubeai/base/cubeai_config.h"
#include "cubeai/io/torch_state_dictionary_reader.h"

#ifdef USE_PYTORCH
namespace cubeai{
namespace torch_utils {


TorchStateDictionaryReader::TorchStateDictionaryReader(const std::string& params_path, bool do_read)
    :
      params_path_(params_path),
      j_()
{
    if(do_read){
        read();
    }
}

void TorchStateDictionaryReader::read(){


}

}
}
#endif
