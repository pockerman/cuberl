#ifndef IRIS_DATA_SET_H
#define IRIS_DATA_SET_H

#include "cubeai/base/cubeai_types.h"
#include <vector>
#include <utility>
#include <map>
#include <string>

namespace cubeai {
namespace datasets {

class IrisDataSet
{
public:

    typedef std::vector<real_t>::value_type value_type;
    typedef std::vector<std::pair<std::vector<real_t>, uint_t>>::iterator iterator;
    typedef std::vector<std::pair<std::vector<real_t>, uint_t>>::const_iterator const_iterator;

    struct IrisMeta
    {
        std::string path;
        bool has_ones;
        std::map<uint_t, std::string> class_map;
    };


    ///
    /// \brief Constructor
    ///
    IrisDataSet(bool do_load=true, bool append_ones=false);

    ///
    /// \brief get_meta
    /// \return
    ///
    const IrisMeta& get_meta()const{return meta_;}

    ///
    /// \brief point
    /// \param idx
    /// \return
    ///
    std::pair<std::vector<real_t>, uint_t> point(uint_t idx)const;

    ///
    /// \brief random_shuffle
    ///
    void random_shuffle();

    ///
    /// \brief load
    ///
    void load(bool append_ones=false);

    iterator begin(){return points_.begin();}
    iterator end(){return points_.end();}

    const_iterator begin()const{return points_.begin();}
    const_iterator end()const{return points_.end();}

private:

    IrisMeta meta_;
    std::vector<std::pair<std::vector<real_t>, uint_t>> points_;

};

inline
std::pair<std::vector<real_t>, uint_t>
IrisDataSet::point(uint_t idx)const{
    return points_[idx];
}

}

}

#endif // IRIS_DATA_SET_H
