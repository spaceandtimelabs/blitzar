#include <iostream>
#include <cassert>
#include <cstdint>

#include <cuda_runtime.h>

static constexpr int bucket_group_size = 255;

class element {
public:
  static constexpr size_t num_limbs_v = 5;

  element() noexcept = default;

  constexpr element(uint64_t x1, uint64_t x2, uint64_t x3, uint64_t x4, uint64_t x5) noexcept
      : data_{x1, x2, x3, x4, x5} {}

private:
  uint64_t data_[num_limbs_v];
};

struct element_p3 {
  element_p3() noexcept = default;

  constexpr element_p3(const element& X, const element& Y, const element& Z,
                       const element& T) noexcept
      : X{X}, Y{Y}, Z{Z}, T{T} {}

  element X;
  element Y;
  element Z;
  element T;

  static constexpr element_p3 identity() noexcept {
    element zero = {0, 0, 0, 0, 0};
    element one = {1, 0, 0, 0, 0};
    return element_p3{
        zero,
        one,
        one,
        zero,
    };
  }
};

template <class T> __global__ void bucket_accumulate(T* bucket_sums) {
  for (int i = 0; i < bucket_group_size; ++i) {
    if constexpr (true) { // change to false and things work
      bucket_sums[i] = T::identity();
    } else {
      auto identity = T::identity();
      bucket_sums[i] = identity;
    }
  }
}

int main() {
  using T = element_p3;

  T* data;
  auto rcode = cudaMalloc(&data, sizeof(T) * bucket_group_size);
  if (rcode != cudaSuccess) {
    std::cerr << "cudaFree failed: " << cudaGetErrorString(rcode) << std::endl;
    return -1;
  }
  bucket_accumulate<T><<<1, 1>>>(data);
  rcode = cudaFree(data);
  if (rcode != cudaSuccess) {
    std::cerr << "cudaFree failed: " << cudaGetErrorString(rcode) << std::endl;
    return -1;
  }
  return 0;
}
