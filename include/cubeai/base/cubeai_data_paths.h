#ifndef CUBEAI_DATA_PATHS_H
#define CUBEAI_DATA_PATHS_H

#include "cubeai/base/cubeai_config.h"
#include <boost/noncopyable.hpp>
#include <string>


namespace cubeai
{

///
/// \brief The CubeAIDataPaths class
///
class CubeAIDataPaths: private boost::noncopyable
{
public:

    ///
    /// \brief CubeAIDataPaths
    ///
    CubeAIDataPaths() = delete;

    ///
    /// \brief data_directory_path
    /// \return
    ///
    static std::string data_directory_path()noexcept{return std::string(DATA_SET_FOLDER);}

    ///
    /// \brief mnist_data_directory_path
    /// \return
    ///
    static std::string mnist_data_directory_path()noexcept{return CubeAIDataPaths::data_directory_path() + "/MNIST/";}

    ///
    /// \brief mnist_data_train_directory_path
    /// \return
    ///
    static std::string mnist_data_train_directory_path()noexcept{return CubeAIDataPaths::mnist_data_directory_path() + "/train/";}

    ///
    /// \brief mnist_data_test_directory_path
    /// \return
    ///
    static std::string mnist_data_test_directory_path()noexcept{return CubeAIDataPaths::mnist_data_directory_path() + "/test/";}

    ///
    /// \brief iris_data_path
    /// \return
    ///
    static std::string iris_data_path()noexcept{return CubeAIDataPaths::data_directory_path() + "/iris_data.csv";}
};
}
#endif // CUBEAI_DATA_PATHS_H
