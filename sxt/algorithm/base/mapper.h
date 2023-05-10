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

namespace sxt::algb {
//--------------------------------------------------------------------------------------------------
// mapper
//--------------------------------------------------------------------------------------------------
/**
 * Describe a generic map function that can be used within CUDA kernels.
 *
 * Mapper turns an index into a value.
 */
template <class M>
concept mapper = requires(M m, typename M::value_type& x, unsigned int i, void* data) {
  { m.map_index(i) } noexcept -> std::convertible_to<typename M::value_type>;
  { m.map_index(x, i) } noexcept;
};
} // namespace sxt::algb
