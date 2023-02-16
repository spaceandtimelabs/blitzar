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
  case block_size_t::v128: {
    return f(std::integral_constant<unsigned int, 128>{});
  }
  case block_size_t::v64: {
    return f(std::integral_constant<unsigned int, 64>{});
  }
  case block_size_t::v32: {
    return f(std::integral_constant<unsigned int, 32>{});
  }
  case block_size_t::v16: {
    return f(std::integral_constant<unsigned int, 16>{});
  }
  case block_size_t::v8: {
    return f(std::integral_constant<unsigned int, 8>{});
  }
  case block_size_t::v4: {
    return f(std::integral_constant<unsigned int, 4>{});
  }
  case block_size_t::v2: {
    return f(std::integral_constant<unsigned int, 2>{});
  }
  case block_size_t::v1: {
    return f(std::integral_constant<unsigned int, 1>{});
  }
  }
  __builtin_unreachable();
}
} // namespace sxt::xenk
