#ifndef DP_ALGO_BASE_H
#define DP_ALGO_BASE_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/io/csv_file_writer.h"
#include "cubeai/rl/algorithms/rl_algorithm_base.h"
//#include "cubeai/rl/algorithms/algorithm_base.h"

#include <tuple>
#include <vector>
#include <string>

namespace cubeai {
namespace rl {
namespace algos {
namespace dp {

///
/// \brief The DPAlgoBase class
///
template<typename EnvType>
class DPAlgoBase: public RLAlgoBase<EnvType>
{
public:

    ///
    /// \brief env_t
    ///
    typedef typename RLAlgoBase<EnvType>::env_type env_type;

    ///
    /// \brief Destructor
    ///
    virtual ~DPAlgoBase() = default;

protected:

    ///
    /// \brief DPAlgoBase
    /// \param name
    ///
    DPAlgoBase()=default;


};

/*template<typename TimeStepTp>
void
DPAlgoBase<TimeStepTp>::save(const std::string& filename)const{

    CSVWriter writer(filename, ',', true);

    std::vector<std::string> columns(2);
    columns[0] = "State Id";
    columns[1] = "Value";
    writer.write_column_names(columns);

    for(uint_t s=0; s < v_.size(); ++s){
        auto row = std::make_tuple(s, v_[s]);
        writer.write_row(row);
    }
}*/


}

}

}

}

#endif // DP_ALGO_BASE_H
