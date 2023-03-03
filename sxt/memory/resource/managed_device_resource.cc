#include "sxt/memory/resource/managed_device_resource.h"

#include <cuda_runtime.h>

#include <string>

#include "sxt/base/error/panic.h"

namespace sxt::memr {
//--------------------------------------------------------------------------------------------------
// do_allocate
//--------------------------------------------------------------------------------------------------
void* managed_device_resource::do_allocate(size_t bytes, size_t /*alignment*/) noexcept {
  void* res;
  auto rcode = cudaMallocManaged(&res, bytes);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMallocManaged failed: " + std::string{cudaGetErrorString(rcode)});
  }
  return res;
}

//--------------------------------------------------------------------------------------------------
// do_deallocate
//--------------------------------------------------------------------------------------------------
void managed_device_resource::do_deallocate(void* ptr, size_t /*bytes*/,
                                            size_t /*alignment*/) noexcept {
  auto rcode = cudaFree(ptr);
  if (rcode != cudaSuccess) {
    baser::panic("cudaFree failed: " + std::string{cudaGetErrorString(rcode)});
  }
}

//--------------------------------------------------------------------------------------------------
// do_is_equal
//--------------------------------------------------------------------------------------------------
bool managed_device_resource::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
  return this == &other;
}

//--------------------------------------------------------------------------------------------------
// get_managed_device_resource
//--------------------------------------------------------------------------------------------------
managed_device_resource* get_managed_device_resource() noexcept {
  // Use a heap allocated object that never gets deleted since we want this
  // to be available for the program duration.
  //
  // See https://google.github.io/styleguide/cppguide.html#Static_and_Global_Variables for
  // details on this use case.
  static managed_device_resource* resource = new managed_device_resource{};

  return resource;
}
} // namespace sxt::memr
