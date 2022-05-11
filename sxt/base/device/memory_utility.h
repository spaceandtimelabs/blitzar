#pragma once

#include <cuda_runtime.h>

#include "sxt/base/device/cuda_utility.h"

namespace sxt::basdv {

//--------------------------------------------------------------------------------------------------
// copy_host_to_device
//--------------------------------------------------------------------------------------------------
void copy_host_to_device(void* dst, const void* src, size_t count) noexcept {
    handle_cuda_error(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice));
}

//--------------------------------------------------------------------------------------------------
// copy_device_to_host
//--------------------------------------------------------------------------------------------------
void copy_device_to_host(void* dst, const void* src, size_t count) noexcept {
    handle_cuda_error(cudaMemcpy(dst, src, count, cudaMemcpyDeviceToHost));
}

//--------------------------------------------------------------------------------------------------
// copy_async_host_to_device
//--------------------------------------------------------------------------------------------------
void copy_async_host_to_device(void* dst, const void* src, size_t count, cudaStream_t stream) noexcept {
    handle_cuda_error(cudaMemcpyAsync(dst, src, count, cudaMemcpyHostToDevice, stream));
}

//--------------------------------------------------------------------------------------------------
// copy_async_device_to_host
//--------------------------------------------------------------------------------------------------
void copy_async_device_to_host(void* dst, const void* src, size_t count, cudaStream_t stream) noexcept {
    handle_cuda_error(cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToHost, stream));
}

}  // namespace sxt::basdv
