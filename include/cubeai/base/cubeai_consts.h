#ifndef CUBEAI_CONSTS_H
#define CUBEAI_CONSTS_H

#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_version.h"
#include <boost/noncopyable.hpp>

#include <string>

namespace cubeai{

///
/// \brief The KernelConsts class. Common quantities used around  the library.
///
class CubeAIConsts: private boost::noncopyable
{

public:

    ///
    /// \brief INVALID_SIZE_TYPE
    ///
    static constexpr uint_t INVALID_SIZE_TYPE = static_cast<uint_t>(-1);

    ///
    /// \brief Returns a string describing the library version
    ///
    static std::string version(){return std::string(CUBEAILIB_VERSION);}

    ///
    /// \brief data_directory_path
    /// \return
    ///
    static std::string data_directory_path()noexcept{return std::string(DATA_SET_FOLDER);}

    ///
    /// \brief mnist_data_directory_path
    /// \return
    ///
    static std::string mnist_data_directory_path()noexcept{return CubeAIConsts::data_directory_path() + "/MNIST/";}

    ///
    /// \brief Returns the value of the tolerance constant.
    /// Default is static_cast<real_type>(1.0e-8)
    ///
    static real_t tolerance(){return tol_;}

    ///
    /// \brief Dummy unit constant indicating the absence of a metric unit
    ///
    static std::string dummy_unit(){return "DUMMY_UNIT";}

    ///
    /// \brief Returns "EOF"
    ///
    static std::string eof_string(){return "EOF";}

    ///
    /// \brief Returns the INFO string
    ///
    static std::string info_str(){return "INFO: ";}

    ///
    /// \brief warning_str Returns the WARNING string
    ///
    static std::string warning_str(){return "WARNING: ";}

    ///
    /// \brief Dummy string
    ///
    static std::string dummy_string(){return "DUMMY_STR";}

    ///
    /// \brief Returns static_cast<uint_t>(-1)
    ///
    static uint_t invalid_size_type(){return static_cast<uint_t>(-1);}

    ///
    /// \brief Initialize the default constants
    ///
    static void initialize();
	
    ///
    /// \brief Constructor
    ///
    CubeAIConsts()=delete;

private:

    ///
    /// \brief The tolerance constant
    ///
    static real_t tol_;
};



}
#endif
