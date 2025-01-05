#ifndef CUBEAI_TYPES_H
#define CUBEAI_TYPES_H

#include "rlenvs/rlenvs_types_v2.h"
#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH
#include <torch/torch.h>
#endif

//#include "blaze/Math.h"
#include "Eigen/Core"
#include "Eigen/Dense"
#include <cstddef>
#include <string>


namespace cuberl
{
	
	using namespace rlenvscpp;
	
	///
	/// \brief int type
	///
	using rlenvscpp::int_t;
	
	///
	/// \brief long int type
	///
	using rlenvscpp::lint_t;

    ///
    /// \brief The float precision type
    ///
	using rlenvscpp::float_t;
    
	///
    /// \brief The double precision type
    ///
	using rlenvscpp::real_t;
    
    ///
    /// \brief General matrix type
    ///
    using rlenvscpp::DynMat; 

	///
	/// \brief General row vector
	///
    using rlenvscpp::DynVec = Eigen::RowVectorX<T>;

    ///
    /// Float type vector
    ///
    using rlenvscpp::FloatVec;
	
	///
	/// Real type vector
	///
	using rlenvscpp::RealVec;

    ///
    /// \brief Null type. Simple placeholder
    ///
    using rlenvscpp::Null{};
	
	///
	/// \brief Device type
	///
	using rlenvscpp::DeviceType;
    
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
