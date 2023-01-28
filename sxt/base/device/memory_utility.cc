#include "sxt/base/device/memory_utility.h"

#include <cuda_runtime.h>

#include <string>

#include "sxt/base/error/panic.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// async_memcpy_host_to_device
//--------------------------------------------------------------------------------------------------
void async_memcpy_host_to_device(void* dst, const void* src, size_t count) noexcept {
  auto rcode = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpyAsync failed: " + std::string(cudaGetErrorString(rcode)));
  }
}

//--------------------------------------------------------------------------------------------------
// memcpy_host_to_device
//--------------------------------------------------------------------------------------------------
void memcpy_host_to_device(void* dst, const void* src, size_t count) noexcept {
  auto rcode = cudaMemcpy(dst, src, count, cudaMemcpyHostToDevice);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpy failed: " + std::string(cudaGetErrorString(rcode)));
  }
}

//--------------------------------------------------------------------------------------------------
// memcpy_device_to_host
//--------------------------------------------------------------------------------------------------
void memcpy_device_to_host(void* dst, const void* src, size_t count) noexcept {
  auto rcode = cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpy failed: " + std::string(cudaGetErrorString(rcode)));
  }
}

//--------------------------------------------------------------------------------------------------
// async_memcpy_host_to_device
//--------------------------------------------------------------------------------------------------
void async_memcpy_host_to_device(void* dst, const void* src, size_t count,
                                 bast::raw_stream_t stream) noexcept {
  auto rcode = cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpyAsync failed: " + std::string(cudaGetErrorString(rcode)));
  }
}

//--------------------------------------------------------------------------------------------------
// async_memcpy_device_to_host
//--------------------------------------------------------------------------------------------------
void async_memcpy_device_to_host(void* dst, const void* src, size_t count,
                                 bast::raw_stream_t stream) noexcept {
  auto rcode = cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemcpyAsync failed: " + std::string(cudaGetErrorString(rcode)));
  }
}

//--------------------------------------------------------------------------------------------------
// memset_device
//--------------------------------------------------------------------------------------------------
void memset_device(void* dst, int value, size_t count) noexcept {
  auto rcode = cudaMemset(dst, value, count);
  if (rcode != cudaSuccess) {
    baser::panic("cudaMemset failed: " + std::string(cudaGetErrorString(rcode)));
  }
}
} // namespace sxt::basdv
