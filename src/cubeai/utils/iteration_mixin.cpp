#include "cubeai/utils/iteration_mixin.h"

namespace cubeai{

 bool
 IterationMixin::continue_iterations() noexcept{

     if(current_iteration_idx_ >= max_iterations_){
           return false;
     }

     current_iteration_idx_++;
     return true;
 }

}
