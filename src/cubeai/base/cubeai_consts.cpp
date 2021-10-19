#include "cubeai/base/cubeai_consts.h"

namespace cubeai
{
    real_t
	CubeAIConsts::tol_ = 1.0e-8;

	void 
	CubeAIConsts::initialize(){

		CubeAIConsts::tol_ = 1.0e-8;
	}

}
