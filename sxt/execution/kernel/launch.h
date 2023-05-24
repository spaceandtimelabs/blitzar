/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#pragma once

#include <type_traits>

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/macro/cuda_warning.h"
#include "sxt/execution/kernel/block_size.h"

namespace sxt::xenk {
//--------------------------------------------------------------------------------------------------
// launch_kernel
//--------------------------------------------------------------------------------------------------
/**
 * Allow us to conveniently launch kernels that take block size as a template parameter.
 */
CUDA_DISABLE_HOSTDEV_WARNING
template <class F> CUDA_CALLABLE void launch_kernel(block_size_t block_size, F f) noexcept {
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
