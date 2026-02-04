#ifndef CUBEAI_TYPES_H
#define CUBEAI_TYPES_H

#include "bitrl/bitrl_types.h"
#include "cuberl/base/cubeai_config.h"

#ifdef USE_PYTORCH
#include <torch/torch.h>
#endif

#include <cstddef>
#include <string>


namespace cuberl
{
	
	using namespace bitrl;
	
	///
	/// \brief int type
	///
	using bitrl::int_t;
	
	///
	/// \brief long int type
	///
	using bitrl::lint_t;

    ///
    /// \brief The float precision type
    ///
	using bitrl::float_t;
    
	///
    /// \brief The double precision type
    ///
	using bitrl::real_t;
    
    ///
    /// \brief General matrix type
    ///
    using bitrl::DynMat;

	///
	/// \brief Square matrix type
	///
	using bitrl::SquareMat;

	///
	/// General fixed size at compile time matrix
	///
	using bitrl::Mat;

	///
	/// \brief General row vector
	///
    using bitrl::DynVec;

    ///
    /// Float type vector
    ///
    using bitrl::FloatVec;
	
	///
	/// Real type vector
	///
	using bitrl::RealVec;

    ///
    /// \brief Null type. Simple placeholder
    ///
    using bitrl::Null;
	
	///
	/// \brief Device type
	///
	using bitrl::DeviceType;
    
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

	

}

#endif
