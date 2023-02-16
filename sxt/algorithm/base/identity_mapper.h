#pragma once

#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::algb {
//--------------------------------------------------------------------------------------------------
// identity_mapper
//--------------------------------------------------------------------------------------------------
template <class T> class identity_mapper {
public:
  using value_type = T;

  identity_mapper() noexcept = default;

  CUDA_CALLABLE explicit identity_mapper(const T* data) noexcept : data_{data} {}

  CUDA_CALLABLE void map_index(T& val, unsigned int index) const noexcept { val = data_[index]; }

  CUDA_CALLABLE const T& map_index(unsigned int index) const noexcept { return data_[index]; }

private:
  const T* data_ = nullptr;
};
} // namespace sxt::algb
