#include "cubeai/base/cubeai_types.h"
#include "cubeai/data_structs/experience_buffer.h"

#include <gtest/gtest.h>
#include <vector>

namespace{

using cubeai::real_t;
using cubeai::uint_t;
using namespace cubeai::containers;

struct Experience
{
    uint_t item;
};
}


TEST(TestA2C, Test_Constructor) {

    ExperienceBuffer<Experience> buffer(3);

    ASSERT_TRUE(buffer.empty());
    ASSERT_EQ(buffer.capacity(), static_cast<uint_t>(3));
    ASSERT_EQ(buffer.size(), static_cast<uint_t>(0));
}

TEST(TestA2C, Test_append) {

    ExperienceBuffer<Experience> buffer(3);

    ASSERT_TRUE(buffer.empty());
    ASSERT_EQ(buffer.capacity(), static_cast<uint_t>(3));

    Experience exp1{1};
    buffer.append(exp1);

    Experience exp2{2};
    buffer.append(exp2);

    Experience exp3{3};
    buffer.append(exp3);

    ASSERT_EQ(buffer.size(), static_cast<uint_t>(3));
    ASSERT_EQ(buffer[0].item, exp1.item);
    ASSERT_EQ(buffer[1].item, exp2.item);
    ASSERT_EQ(buffer[2].item, exp3.item);
}

TEST(TestA2C, Test_replacement) {

    ExperienceBuffer<Experience> buffer(3);

    ASSERT_TRUE(buffer.empty());
    ASSERT_EQ(buffer.capacity(), static_cast<uint_t>(3));

    Experience exp1{1};
    buffer.append(exp1);

    Experience exp2{2};
    buffer.append(exp2);

    Experience exp3{3};
    buffer.append(exp3);

    ASSERT_EQ(buffer.size(), static_cast<uint_t>(3));
    ASSERT_EQ(buffer[0].item, exp1.item);
    ASSERT_EQ(buffer[1].item, exp2.item);
    ASSERT_EQ(buffer[2].item, exp3.item);

    // now add an extra element
    Experience exp4{4};
    buffer.append(exp4);

    ASSERT_EQ(buffer[0].item, exp2.item);
    ASSERT_EQ(buffer[1].item, exp3.item);
    ASSERT_EQ(buffer[2].item, exp4.item);


}



