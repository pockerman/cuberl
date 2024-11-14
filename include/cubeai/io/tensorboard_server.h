// SPDX-FileCopyrightText: 2024 <copyright holder> <email>
// SPDX-License-Identifier: Apache-2.0

#ifndef TENSORBOARD_SERVER_H
#define TENSORBOARD_SERVER_H


#include "cubeai/base/cubeai_types.h"
#include "cubeai/base/cubeai_consts.h"

#include <boost/noncopyable.hpp>
#include <string>
#include <string_view>
#include <unordered_map>
namespace cubeai{
namespace io {
/**
 * @todo
 */
class TensorboardServer: private boost::noncopyable
{
public:



    /**
     * @brief Constructor
     **/
    TensorboardServer(const std::string& api_url );

    /**
     * @brief Set the log_dir for this instance
     **/
    void init(const std::string& log_dir);

    /**
     * @brief Close the TensorboardServer
     */
    void close();

    /**
     * @brief Add scalar
     */
    void add_scalar(const std::string& tag, real_t value,
                    uint_t step_idx=CubeAIConsts::invalid_size_type())const;

    /**
     * @brief Add scalar
     */
    void add_scalars(const std::string& main_tag,
                     const std::unordered_map<std::string, real_t>& values,
                     uint_t step_idx=CubeAIConsts::invalid_size_type())const;

     /**
     * @brief Add scalar
     */
    void add_text(const std::string& tag,
                  const std::string& text,
                  uint_t step_idx=CubeAIConsts::invalid_size_type())const;

    /**
     * @brief Return the directory for logging
     *
     * */
    std::string_view get_log_dir_path()const noexcept{return log_dir_;}


    /**
     * @brief Return the server address
     *
     */
    std::string_view get_server_address()const noexcept {return api_url_;}



private:

    std::string api_url_;
    std::string log_dir_;

    const std::string api_str_;


};

}
}
#endif // TENSORBOARD_SERVER_H
