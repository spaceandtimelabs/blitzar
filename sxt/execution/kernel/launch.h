#pragma once

#include <type_traits>

#include "sxt/execution/kernel/block_size.h"

namespace sxt::xenk {
//--------------------------------------------------------------------------------------------------
// launch_kernel
//--------------------------------------------------------------------------------------------------
/**
 * Allow us to conveniently launch kernels that take block size as a template parameter.
 */
template <class F> void launch_kernel(block_size_t block_size, F f) noexcept {
  switch (block_size) {
  case block_size_t::v32: {
    f(std::integral_constant<unsigned int, 32>{});
    break;
  }
  case block_size_t::v64: {
    f(std::integral_constant<unsigned int, 64>{});
    break;
  }
  case block_size_t::v128: {
    f(std::integral_constant<unsigned int, 128>{});
    break;
  }
  }
}
} // namespace sxt::xenk
