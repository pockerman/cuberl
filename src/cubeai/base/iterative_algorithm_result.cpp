#include "cubeai/base/iterative_algorithm_result.h"

namespace cubeai{

IterativeAlgorithmResult::IterativeAlgorithmResult(real_t tol, real_t res,
                                                   uint_t nitrs, std::chrono::seconds time)
    :
      tolerance(tol),
      residual(res),
      num_iterations(nitrs),
      total_time(time)
{}

std::ostream&
IterativeAlgorithmResult:: print(std::ostream& out)const{

    out<<"Converged...: "<<std::boolalpha<<converged<<std::endl;
    out<<"Tolerance...: "<<tolerance<<std::endl;
    out<<"Residual....: "<<residual<<std::endl;
    out<<"Iterations..: "<<num_iterations<<std::endl;
    out<<"Total time..: "<<total_time.count()<<std::endl;
    return out;
}
}
