#pragma once

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/type/raw_stream.h"

namespace sxt::algb {
//--------------------------------------------------------------------------------------------------
// identity_mapper
//--------------------------------------------------------------------------------------------------
template <class T> class identity_mapper {
public:
  using value_type = T;
  static constexpr auto num_bytes_per_index = sizeof(T);

  identity_mapper() noexcept = default;

  CUDA_CALLABLE explicit identity_mapper(const T* data) noexcept : data_{data} {}

  CUDA_CALLABLE void map_index(T& val, unsigned int index) const noexcept { val = data_[index]; }

  CUDA_CALLABLE const T& map_index(unsigned int index) const noexcept { return data_[index]; }

  identity_mapper async_make_host_mapper(void* host_data, bast::raw_stream_t stream, unsigned int n,
                                         unsigned int offset) const noexcept {
    SXT_DEBUG_ASSERT(offset < n);
    basdv::async_memcpy_device_to_host(host_data, data_ + offset,
                                       (n - offset) * num_bytes_per_index, stream);
    return identity_mapper{reinterpret_cast<T*>(host_data)};
  }

private:
  const T* data_ = nullptr;
};
} // namespace sxt::algb
