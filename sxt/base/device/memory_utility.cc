#include "sxt/base/device/memory_utility.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// async_memcpy_host_to_device
//--------------------------------------------------------------------------------------------------
void async_memcpy_host_to_device(void* dst, const void* src, size_t count) noexcept {
  auto rcode = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice);
  if (rcode != cudaSuccess) {
    std::cerr << "cudaMemcpyAsync failed: " << cudaGetErrorString(rcode) << "\n";
    std::abort();
  }
}

//--------------------------------------------------------------------------------------------------
// memcpy_host_to_device
//--------------------------------------------------------------------------------------------------
void memcpy_host_to_device(void* dst, const void* src, size_t count) noexcept {
  auto rcode = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
  if (rcode != cudaSuccess) {
    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(rcode) << "\n";
    std::abort();
  }
}

//--------------------------------------------------------------------------------------------------
// memcpy_device_to_host
//--------------------------------------------------------------------------------------------------
void memcpy_device_to_host(void* dst, const void* src, size_t count) noexcept {
  auto rcode = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
  if (rcode != cudaSuccess) {
    std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(rcode) << "\n";
    std::abort();
  }
}

//--------------------------------------------------------------------------------------------------
// async_memcpy_host_to_device
//--------------------------------------------------------------------------------------------------
void async_memcpy_host_to_device(void* dst, const void* src, size_t count,
                                 cudaStream_t stream) noexcept {
  auto rcode = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);
  if (rcode != cudaSuccess) {
    std::cerr << "cudaMemcpyAsync failed: " << cudaGetErrorString(rcode) << "\n";
    std::abort();
  }
}

//--------------------------------------------------------------------------------------------------
// async_memcpy_device_to_host
//--------------------------------------------------------------------------------------------------
void async_memcpy_device_to_host(void* dst, const void* src, size_t count,
                                 cudaStream_t stream) noexcept {
  auto rcode = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream);
  if (rcode != cudaSuccess) {
    std::cerr << "cudaMemcpyAsync failed: " << cudaGetErrorString(rcode) << "\n";
    std::abort();
  }
}
} // namespace sxt::basdv
