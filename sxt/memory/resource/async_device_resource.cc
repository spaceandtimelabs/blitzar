#include "sxt/memory/resource/async_device_resource.h"

#include <cuda_runtime.h>

#include <string>

#include "sxt/base/error/panic.h"

namespace sxt::memr {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
async_device_resource::async_device_resource(bast::raw_stream_t stream) noexcept
    : stream_{stream} {}

//--------------------------------------------------------------------------------------------------
// do_allocate
//--------------------------------------------------------------------------------------------------
void* async_device_resource::do_allocate(size_t bytes, size_t alignment) noexcept {
  void* res;
  auto rcode = cudaMallocAsync(&res, bytes, stream_);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMallocAsync failed: " + std::string{cudaGetErrorString(rcode)});
  }
  return res;
}

//--------------------------------------------------------------------------------------------------
// do_deallocate
//--------------------------------------------------------------------------------------------------
void async_device_resource::do_deallocate(void* ptr, size_t bytes, size_t alignment) noexcept {
  auto rcode = cudaFreeAsync(ptr, stream_);
  if (rcode != cudaSuccess) {
    baser::panic("cudaFreeAsync failed: " + std::string{cudaGetErrorString(rcode)});
  }
}

//--------------------------------------------------------------------------------------------------
// do_is_equal
//--------------------------------------------------------------------------------------------------
bool async_device_resource::do_is_equal(const std::pmr::memory_resource& other) const noexcept {
  return this == &other;
}
} // namespace sxt::memr
