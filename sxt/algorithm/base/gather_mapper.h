#pragma once

#include "sxt/base/error/assert.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::algb {
//--------------------------------------------------------------------------------------------------
// gather_mapper
//--------------------------------------------------------------------------------------------------
template <class T> class gather_mapper {
public:
  using value_type = T;

  gather_mapper() noexcept = default;

  CUDA_CALLABLE gather_mapper(const T* data, const unsigned* indexes) noexcept
      : data_{data}, indexes_{indexes} {}

  CUDA_CALLABLE void map_index(T& val, unsigned int index) const noexcept {
    val = data_[indexes_[index]];
  }

  CUDA_CALLABLE const T& map_index(unsigned int index) const noexcept {
    return data_[indexes_[index]];
  }

private:
  const T* data_ = nullptr;
  const unsigned* indexes_ = nullptr;
};
} // namespace sxt::algb
