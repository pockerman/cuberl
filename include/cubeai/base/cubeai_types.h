#ifndef CUBEAI_TYPES_H
#define CUBEAI_TYPES_H

#include "cubeai/base/cubeai_config.h"

#ifdef USE_PYTORCH
#include <torch/torch.h>
#endif

#include "blaze/Math.h"
#include <cstddef>
#include <string>


namespace cubeai
{
    ///
    /// \brief unsigned int type
    ///
    typedef std::size_t uint_t;

    ///
    /// \brief Configure the double precision type
    ///
    #ifdef CUBEAI_REAL_TYPE_FLOAT
        typedef float real_t;
    #else
        typedef double real_t;
    #endif

    ///
    /// \brief General matrix type
    ///
    template<typename T>
    using DynMat = blaze::DynamicMatrix<T, blaze::rowMajor>;

    ///
    /// \brief General diagonal matrix
    ///
    template<typename T>
    using DiagMat = blaze::DiagonalMatrix<DynMat<T>>;

    ///
    /// \brief Identity matrix
    ///
    template<typename T>
    using IdentityMatrix = blaze::IdentityMatrix<T>;

    ///
    /// \brief General Sparse matrix
    ///
    template<typename T>
    using SparseMatrix = blaze::CompressedMatrix<T, blaze::rowMajor>;

    ///
    /// \brief General vector type. By default this is
    /// a column vector
    ///
    template<typename T>
    using DynVec = blaze::DynamicVector<T, blaze::columnVector>;

    ///
    ///
    ///
    using FloatVec = DynVec<real_t>;

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
#endif

}

#endif
