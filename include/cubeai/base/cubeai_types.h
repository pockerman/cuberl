#ifndef CUBEAI_TYPES_H
#define CUBEAI_TYPES_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH
#include <torch/torch.h>
#endif

//#include "blaze/Math.h"
#include "Eigen/Core"
#include <cstddef>
#include <string>


namespace cubeai
{
    ///
    /// \brief unsigned int type
    ///
    typedef std::size_t uint_t;
	
	///
	/// \brief int type
	///
	typedef int int_t;
	
	///
	/// \brief long int type
	///
	typedef long int lint_t;

    ///
    /// \brief The float precision type
    ///
    
	typedef float float_t;
    
	///
    /// \brief The double precision type
    ///
	typedef double real_t;
    

    ///
    /// \brief General matrix type
    ///
    template<typename T>
    using DynMat = Eigen::MatrixX<T>;


    template<typename T>
    using DynVec = Eigen::RowVectorX<T>;

    ///
    /// \brief Null type. Simple placeholder
    ///
    struct Null{};
    
#ifdef USE_PYTORCH
   ///
   /// \brief placeholder for torch
   ///
   struct torch_t {};

   ///
   /// \brief torch_int_t
   ///
   typedef long int torch_int_t;

   ///
   /// \brief torch_tensor_t
   ///
   typedef torch::Tensor torch_tensor_t;
   
   
	template<typename T>
	struct torch_tensor_value_type{
		typedef T value_type;
	};
#endif

	///
	/// \brief Device type
	///
	enum class DeviceType {INVALID_TYPE=0, CPU=1, GPU=2} ;

}

#endif
