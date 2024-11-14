#include "cubeai/maths/vector_math.h"
#include <cmath>
namespace cubeai{
namespace maths{

std::vector<real_t>
logspace(real_t start, real_t end, uint_t num, real_t base){

    std::vector<real_t> logspace(num, 0.0);
    logspace[0] = std::pow(base, start);
    logspace[num - 1 ] = std::pow(base, end);
    real_t dx = (end - start) / static_cast<real_t>(num - 1);

    for (int i = 1; i < num - 1; ++i){
        auto point = start + i*dx;
        logspace[i] = std::pow(base, point);
    }

    return logspace;

}

}
}
