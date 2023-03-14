#include "sxt/base/device/memory_utility.h"

#include <cuda_runtime.h>

#include <memory>
#include <vector>

#include "sxt/base/container/span.h"
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

  SECTION("we can copy memory") {
    std::vector<int> v = {1, 2, 3};
    std::vector<int> w(v.size());
    int* data;
    REQUIRE(cudaMalloc(&data, sizeof(int) * v.size()) == cudaSuccess);
    ;
    basct::span<int> buffer{data, v.size()};
    cudaStream_t stream;
    REQUIRE(cudaStreamCreate(&stream) == cudaSuccess);
    async_copy_host_to_device(buffer, v, stream);
    async_copy_device_to_host(w, buffer, stream);
    REQUIRE(cudaStreamSynchronize(stream) == cudaSuccess);
    REQUIRE(w == v);
    REQUIRE(cudaStreamDestroy(stream) == cudaSuccess);
    REQUIRE(cudaFree(data) == cudaSuccess);
  }
}
