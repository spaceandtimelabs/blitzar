#include "sxt/base/device/memory_utility.h"

#include <cuda_runtime.h>

#include <memory>

#include "sxt/base/test/unit_test.h"

using namespace sxt;
using namespace sxt::basdv;

TEST_CASE("we can determine if a pointer is device or host") {
  SECTION("we handle host-side pointers") {
    auto ptr = std::make_unique<int>(123);
    REQUIRE(!is_device_pointer(ptr.get()));
  }

  SECTION("we handle device pointers") {
    void* ptr;
    REQUIRE(cudaMalloc(&ptr, 100) == cudaSuccess);
    REQUIRE(is_device_pointer(ptr));
    cudaFree(ptr);
  }

  SECTION("we handle managed pointers") {
    void* ptr;
    REQUIRE(cudaMallocManaged(&ptr, 100) == cudaSuccess);
    REQUIRE(is_device_pointer(ptr));
    cudaFree(ptr);
  }
}
