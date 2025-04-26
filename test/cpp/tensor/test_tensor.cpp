#include <gtest/gtest.h>
#include "core/tensor/tensor.h"

namespace torchscratch::core::tensor {

TEST(TensorTest, ConstructorAndAccessors) {
  // Test shape-based constructor
  Tensor t({2, 3});
  EXPECT_EQ(t.dim(), 2);
  EXPECT_EQ(t.numel(), 6);
  EXPECT_EQ(t.shape(), std::vector<int64_t>({2, 3}));
  EXPECT_EQ(t.strides(), std::vector<int64_t>({3, 1}));
  EXPECT_FALSE(t.data_ptr()); // No data allocated yet
}

TEST(TensorTest, AllocateAndDeallocate) {
  Tensor t({2, 2});
  EXPECT_FALSE(t.data_ptr());
  t.allocate();
  EXPECT_TRUE(t.data_ptr());
  EXPECT_TRUE(t.is_contiguous());
  t.deallocate();
  EXPECT_FALSE(t.data_ptr());
  EXPECT_FALSE(t.is_contiguous());
}

TEST(TensorTest, Reshape) {
  Tensor t({2, 3});
  t.allocate();
  Tensor reshaped = t.reshape({3, 2});
  EXPECT_EQ(reshaped.shape(), std::vector<int64_t>({3, 2}));
  EXPECT_EQ(reshaped.numel(), 6);
  EXPECT_FALSE(reshaped.is_contiguous());
  EXPECT_EQ(reshaped.data_ptr(), t.data_ptr()); // Shallow copy

  // Test invalid reshape
  EXPECT_THROW(t.reshape({2, 4}), std::runtime_error);
}

TEST(TensorTest, Clone) {
  Tensor t({2, 2});
  t.allocate();
  Tensor cloned = t.clone();
  EXPECT_EQ(cloned.shape(), t.shape());
  EXPECT_EQ(cloned.numel(), t.numel());
  EXPECT_TRUE(cloned.is_contiguous());
  EXPECT_NE(cloned.data_ptr(), t.data_ptr()); // Deep copy
}

TEST(TensorTest, CopyAndMove) {
  Tensor t1({2, 3});
  t1.allocate();

  // Copy constructor
  Tensor t2(t1);
  EXPECT_EQ(t2.shape(), t1.shape());
  EXPECT_EQ(t2.data_ptr(), t1.data_ptr()); // Shared data

  // Move constructor
  Tensor t3(std::move(t1));
  EXPECT_EQ(t3.shape(), std::vector<int64_t>({2, 3}));
  EXPECT_FALSE(t1.data_ptr()); // t1 is moved
}

} // namespace torchscratch::core::tensor

int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}