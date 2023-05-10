/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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

#include <concepts>

#include "sxt/algorithm/base/mapper.h"
#include "sxt/algorithm/base/reducer.h"
#include "sxt/base/macro/cuda_callable.h"

namespace sxt::algb {
//--------------------------------------------------------------------------------------------------
// accumulate
//--------------------------------------------------------------------------------------------------
template <algb::reducer Reducer, algb::mapper Mapper>
  requires std::same_as<typename Reducer::value_type, typename Mapper::value_type>
CUDA_CALLABLE void accumulate(typename Reducer::value_type& res, typename Reducer::value_type& e,
                              Mapper mapper, unsigned int i) noexcept
  requires(!requires { Reducer::accumulate_inplace(res, e, mapper, i); })
{
  mapper.map_index(e, i);
  Reducer::accumulate_inplace(res, e);
}

template <algb::reducer Reducer, algb::mapper Mapper>
  requires std::same_as<typename Reducer::value_type, typename Mapper::value_type>
CUDA_CALLABLE void accumulate(typename Reducer::value_type& res, typename Reducer::value_type& e,
                              Mapper mapper, unsigned int i) noexcept
  requires(requires { Reducer::accumulate_inplace(res, e, mapper, i); })
{
  // support specialized accumulate that might be more efficient
  Reducer::accumulate_inplace(res, e, mapper, i);
}
} // namespace sxt::algb
