#include "sxt/memory/resource/device_resource.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace sxt::memr {
//--------------------------------------------------------------------------------------------------
// do_allocate
//--------------------------------------------------------------------------------------------------
void* device_resource::do_allocate(size_t bytes, size_t /*alignment*/) noexcept {
  void* res;
  auto rcode = cudaMalloc(&res, bytes);
  if (rcode != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << rcode << "\n";
    std::abort();
  }
  return res;
}

//--------------------------------------------------------------------------------------------------
// do_deallocate
//--------------------------------------------------------------------------------------------------
void device_resource::do_deallocate(void* ptr, size_t /*bytes*/, size_t /*alignment*/) noexcept {
  auto rcode = cudaFree(ptr);
  if (rcode != cudaSuccess) {
    std::cerr << "cudaFree failed: " << rcode << "\n";
    std::abort();
  }
}

//--------------------------------------------------------------------------------------------------
// do_is_equal
//--------------------------------------------------------------------------------------------------
bool device_resource::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
  return this == &other;
}

//--------------------------------------------------------------------------------------------------
// get_device_resource
//--------------------------------------------------------------------------------------------------
device_resource* get_device_resource() noexcept {
  // Use a heap allocated object that never gets deleted since we want this
  // to be available for the program duration.
  //
  // See https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables for
  // details on this use case.
  static device_resource* resource = new device_resource{};

  return resource;
}
} // namespace sxt::memr
