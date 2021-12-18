#ifndef VECTOR_EXPERIENCE_BUFFER_H
#define VECTOR_EXPERIENCE_BUFFER_H

#include "cubeai/base/cubeai_types.h"

#include "boost/noncopyable.hpp"
#include <memory>
#include <vector>
#include <type_traits>

namespace cubeai{
namespace rl{

template<typename ExperienceType, class AllocatorTp = std::allocator<ExperienceType>>
class VectorExperienceBuffer: private boost::noncopyable
{
public:

    static_assert(std::is_default_constructible<ExperienceType>::value, "The experience type is not default constructible");

    typedef ExperienceType value_type;

    typedef AllocatorTp allocator_type ;

    typedef typename std::vector<value_type, allocator_type>::iterator iterator;
    typedef typename std::vector<value_type, allocator_type>::const_iterator const_iterator;
    typedef typename std::vector<value_type, allocator_type>::reverse_iterator reverse_iterator;
    typedef typename std::vector<value_type, allocator_type>::const_reverse_iterator const_reverse_iterator;

    ///
    ///
    ///
    VectorExperienceBuffer()=default;

    ///
    ///
    ///
    explicit VectorExperienceBuffer(uint_t size);

    ///
    /// \brief size
    /// \return
    ///
    uint_t size()const noexcept{return memory_.size();}

    ///
    /// \brief empty
    /// \return
    ///
    bool empty()const noexcept{return memory_.empty();}

    ///
    /// \brief clear
    ///
    void clear()noexcept{memory_.clear();}

    ///
    /// \brief capacity
    /// \return
    ///
    uint_t capacity()const noexcept{return memory_.capacity();}

    ///
    /// \brief sample. Sample batch_size experiences from the
    /// buffer and transfer them in the BatchTp container.
    ///
    template<typename BatchTp>
    void sample(uint_t batch_size, BatchTp& batch)const;

    ///
    /// \brief append Add the experience item in the buffer
    ///
    void append(const value_type& experience){memory_.emplace_back(experience);}

    ///
    /// \brief begin
    /// \return
    ///
    iterator begin(){return memory_.begin();}

    ///
    /// \brief end
    /// \return
    ///
    iterator end(){return memory_.end();}

    ///
    /// \brief begin
    /// \return
    ///
    const_iterator begin()const{return memory_.begin();}

    ///
    /// \brief end
    /// \return
    ///
    const_iterator end()const{return memory_.end();}

    ///
    /// \brief begin
    /// \return
    ///
    reverse_iterator rbegin(){return memory_.rbegin();}

    ///
    /// \brief end
    /// \return
    ///
    reverse_iterator rend(){return memory_.rend();}

    ///
    /// \brief begin
    /// \return
    ///
    const_iterator rbegin()const{return memory_.rbegin();}

    ///
    /// \brief end
    /// \return
    ///
    const_iterator rend()const{return memory_.rend();}

private:

    std::vector<value_type, allocator_type> memory_;

};

template<typename ExperienceType, typename AllocatorTp>
VectorExperienceBuffer<ExperienceType, AllocatorTp>::VectorExperienceBuffer(uint_t size)
    :
      memory_(size)
{}

template<typename ExperienceType, typename AllocatorTp>
template<typename BatchTp>
void
VectorExperienceBuffer<ExperienceType, AllocatorTp>::sample(uint_t batch_size, BatchTp& batch)const{

throw std::logic_error("This function is not implemented!!!");
}
}
}
#endif // VECTOR_EXPERIENCE_BUFFER_H
