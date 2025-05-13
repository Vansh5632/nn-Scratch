#include <gtest/gtest.h>

#include "core/tensor/ops.h"  // Include tensor operations
#include "core/tensor/tensor.h"

namespace torchscratch::core::tensor {

TEST(TensorTest, ConstructorAndAccessors) {
  Tensor t({2, 3});
  EXPECT_EQ(t.dim(), 2);
  EXPECT_EQ(t.numel(), 6);
  EXPECT_EQ(t.shape(), std::vector<int64_t>({2, 3}));
  EXPECT_EQ(t.strides(), std::vector<int64_t>({3, 1}));
  EXPECT_FALSE(t.data_ptr());
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
  EXPECT_EQ(reshaped.data_ptr(), t.data_ptr());

  EXPECT_THROW(t.reshape({2, 4}), std::runtime_error);
}

TEST(TensorTest, Clone) {
  Tensor t({2, 2});
  t.allocate();
  Tensor cloned = t.clone();
  EXPECT_EQ(cloned.shape(), t.shape());
  EXPECT_EQ(cloned.numel(), t.numel());
  EXPECT_TRUE(cloned.is_contiguous());
  EXPECT_NE(cloned.data_ptr(), t.data_ptr());
}

TEST(TensorTest, CopyAndMove) {
  Tensor t1({2, 3});
  t1.allocate();

  Tensor t2(t1);
  EXPECT_EQ(t2.shape(), t1.shape());
  EXPECT_EQ(t2.data_ptr(), t1.data_ptr());

  Tensor t3(std::move(t1));
  EXPECT_EQ(t3.shape(), std::vector<int64_t>({2, 3}));
  EXPECT_FALSE(t1.data_ptr());
}

TEST(TensorTest, Add) {
  Tensor a({2, 2});
  Tensor b({2, 2});
  a.allocate();
  b.allocate();

  float* a_data = a.data_ptr<float>();
  float* b_data = b.data_ptr<float>();
  for (int i = 0; i < 4; ++i) {
    a_data[i] = static_cast<float>(i);
    b_data[i] = static_cast<float>(i + 1);
  }

  Tensor c = add(a, b);
  float* c_data = c.data_ptr<float>();
  EXPECT_EQ(c.shape(), std::vector<int64_t>({2, 2}));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(c_data[i], a_data[i] + b_data[i]);
  }

  // Test broadcasting
  Tensor scalar({1});
  scalar.allocate();
  scalar.data_ptr<float>()[0] = 2.0f;
  Tensor d = add(a, scalar);
  float* d_data = d.data_ptr<float>();
  EXPECT_EQ(d.shape(), std::vector<int64_t>({2, 2}));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(d_data[i], a_data[i] + 2.0f);
  }
}

TEST(TensorTest, Mul) {
  Tensor a({2, 2});
  Tensor b({2, 2});
  a.allocate();
  b.allocate();

  float* a_data = a.data_ptr<float>();
  float* b_data = b.data_ptr<float>();
  for (int i = 0; i < 4; ++i) {
    a_data[i] = static_cast<float>(i);
    b_data[i] = static_cast<float>(i + 1);
  }

  Tensor c = mul(a, b);
  float* c_data = c.data_ptr<float>();
  EXPECT_EQ(c.shape(), std::vector<int64_t>({2, 2}));
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(c_data[i], a_data[i] * b_data[i]);
  }
}

TEST(TensorTest, Matmul) {
  Tensor a({2, 3});
  Tensor b({3, 2});
  a.allocate();
  b.allocate();

  float* a_data = a.data_ptr<float>();
  float* b_data = b.data_ptr<float>();
  for (int i = 0; i < 6; ++i) {
    a_data[i] = static_cast<float>(i);
    b_data[i] = static_cast<float>(i + 1);
  }

  Tensor c = matmul(a, b);
  EXPECT_EQ(c.shape(), std::vector<int64_t>({2, 2}));
  float* c_data = c.data_ptr<float>();
  EXPECT_FLOAT_EQ(c_data[0], 13.0f);
  EXPECT_FLOAT_EQ(c_data[1], 16.0f);
  EXPECT_FLOAT_EQ(c_data[2], 40.0f);
  EXPECT_FLOAT_EQ(c_data[3], 52.0f);

  // Test invalid shapes
  Tensor d({2, 2});
  EXPECT_THROW(matmul(a, d), std::runtime_error);
}

TEST(TensorTest, Transpose) {
  Tensor a({2, 3});
  a.allocate();
  float* a_data = a.data_ptr<float>();
  for (int i = 0; i < 6; ++i) {
    a_data[i] = static_cast<float>(i);
  }

  Tensor b = transpose(a, 0, 1);
  EXPECT_EQ(b.shape(), std::vector<int64_t>({3, 2}));
  EXPECT_FALSE(b.is_contiguous());
  EXPECT_EQ(b.data_ptr(), a.data_ptr());  // Shallow copy

  // Use the transposed accessor instead of raw data pointer
  auto b_accessor = b.transposed_accessor<float>();
  EXPECT_FLOAT_EQ(b_accessor[0], 0.0f);
  EXPECT_FLOAT_EQ(b_accessor[1], 3.0f);
  EXPECT_FLOAT_EQ(b_accessor[2], 1.0f);
  EXPECT_FLOAT_EQ(b_accessor[3], 4.0f);

  EXPECT_THROW(transpose(a, -1, 0), std::runtime_error);
}

}  // namespace torchscratch::core::tensor

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}