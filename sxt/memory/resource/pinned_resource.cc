#include "sxt/memory/resource/pinned_resource.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace sxt::memr {
//--------------------------------------------------------------------------------------------------
// do_allocate
//--------------------------------------------------------------------------------------------------
void* pinned_resource::do_allocate(size_t bytes, size_t /*alignment*/) noexcept {
  void* res;
  auto rcode = cudaMallocHost(&res, bytes);
  if (rcode != cudaSuccess) {
    std::cerr << "cudaMalloc failed: " << rcode << "\n";
    std::abort();
  }
  return res;
}

//--------------------------------------------------------------------------------------------------
// do_deallocate
//--------------------------------------------------------------------------------------------------
void pinned_resource::do_deallocate(void* ptr, size_t /*bytes*/, size_t /*alignment*/) noexcept {
  auto rcode = cudaFreeHost(ptr);
  if (rcode != cudaSuccess) {
    std::cerr << "cudaFree failed: " << rcode << "\n";
    std::abort();
  }
}

//--------------------------------------------------------------------------------------------------
// do_is_equal
//--------------------------------------------------------------------------------------------------
bool pinned_resource::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
  return this == &other;
}

//--------------------------------------------------------------------------------------------------
// get_pinned_resource
//--------------------------------------------------------------------------------------------------
pinned_resource* get_pinned_resource() noexcept {
  // Use a heap allocated object that never gets deleted since we want this
  // to be available for the program duration.
  //
  // See https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables for
  // details on this use case.
  static pinned_resource* resource = new pinned_resource{};

  return resource;
}
} // namespace sxt::memr
