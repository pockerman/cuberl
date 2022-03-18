#include "cubeai/base/cubeai_types.h"
#include "cubeai/data_structs/fixed_size_priority_queue.h"

#include <gtest/gtest.h>
#include <vector>

namespace{

using cubeai::real_t;
using cubeai::uint_t;

}


TEST(TestFixedSizeMaxPriorityQueue, Test_constructor) {

    cubeai::containers::FixedSizeMaxPriorityQueue<uint_t> priority(3);
    ASSERT_TRUE(priority.empty());
    ASSERT_EQ(priority.capacity(), static_cast<uint_t>(3));
}

TEST(TestFixedSizeMaxPriorityQueue, Test_push) {

    cubeai::containers::FixedSizeMaxPriorityQueue<uint_t> priority(3);
    ASSERT_TRUE(priority.empty());
    ASSERT_EQ(priority.capacity(), static_cast<uint_t>(3));

    priority.push(1);
    priority.push(2);
    priority.push(4);

    const auto& item = priority.top();
    ASSERT_EQ(item, static_cast<uint_t>(4));

}


TEST(TestFixedSizeMaxPriorityQueue, Test_push_and_pop) {

    cubeai::containers::FixedSizeMaxPriorityQueue<uint_t> priority(3);
    ASSERT_TRUE(priority.empty());
    ASSERT_EQ(priority.capacity(), static_cast<uint_t>(3));

    priority.push(1);
    priority.push(2);
    priority.push(4);

    const auto& item = priority.top();
    ASSERT_EQ(item, static_cast<uint_t>(4));

    priority.push(5);

    ASSERT_EQ(priority.size(), static_cast<uint_t>(3));
    const auto& item_top = priority.top();
    ASSERT_EQ(item_top, static_cast<uint_t>(5));

    priority.pop();
    ASSERT_EQ(priority.top(), static_cast<uint_t>(4));
}


TEST(TestFixedSizeMinPriorityQueue, Test_constructor) {

    cubeai::containers::FixedSizeMinPriorityQueue<uint_t> priority(3);
    ASSERT_TRUE(priority.empty());
    ASSERT_EQ(priority.capacity(), static_cast<uint_t>(3));
}

TEST(TestFixedSizeMinPriorityQueue, Test_push) {

    cubeai::containers::FixedSizeMinPriorityQueue<uint_t> priority(3);
    ASSERT_TRUE(priority.empty());
    ASSERT_EQ(priority.capacity(), static_cast<uint_t>(3));

    priority.push(1);
    priority.push(2);
    priority.push(4);

    const auto& item = priority.top();
    ASSERT_EQ(item, static_cast<uint_t>(1));

}


TEST(TestFixedSizeMinPriorityQueue, Test_push_and_pop) {

    cubeai::containers::FixedSizeMinPriorityQueue<uint_t> priority(3);
    ASSERT_TRUE(priority.empty());
    ASSERT_EQ(priority.capacity(), static_cast<uint_t>(3));

    priority.push(1);
    priority.push(2);
    priority.push(4);

    const auto& item = priority.top();
    ASSERT_EQ(item, static_cast<uint_t>(1));

    priority.push(0);

    ASSERT_EQ(priority.size(), static_cast<uint_t>(3));
    const auto& item_top = priority.top();
    ASSERT_EQ(item_top, static_cast<uint_t>(0));
}

