/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
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
#include <memory>
#include <vector>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/memory/alloc.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor.h"
#include "sxt/multiexp/pippenger2/partition_table.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// make_in_memory_partition_table_accessor_impl
//--------------------------------------------------------------------------------------------------
template <class U, bascrv::element T>
  requires requires(const U& u, const T& e) {
    static_cast<U>(e);
    T{u};
  }
std::unique_ptr<partition_table_accessor<U>>
make_in_memory_partition_table_accessor_impl(basct::cspan<T> generators, basm::alloc_t alloc,
                                             unsigned window_width = 15) noexcept {
  auto n = generators.size();
  auto partition_table_size = 1u << window_width;
  std::vector<T> generators_data;
  auto num_partitions = basn::divide_up(n, size_t{window_width});
  if (n % window_width != 0) {
    n = num_partitions * window_width;
    generators_data.resize(n);
    auto iter = std::copy(generators.begin(), generators.end(), generators_data.begin());
    std::fill(iter, generators_data.end(), T::identity());
    generators = generators_data;
  }
  memmg::managed_array<U> sums{partition_table_size * num_partitions, alloc};
  compute_partition_table<U, T>(sums, window_width, generators);
  return std::make_unique<in_memory_partition_table_accessor<U>>(std::move(sums), window_width);
}

//--------------------------------------------------------------------------------------------------
// make_in_memory_partition_table_accessor
//--------------------------------------------------------------------------------------------------
template <class U, class T>
  requires(!std::same_as<U, T>)
std::unique_ptr<partition_table_accessor<U>>
make_in_memory_partition_table_accessor(basct::cspan<T> generators,
                                        basm::alloc_t alloc = memr::get_pinned_resource(),
                                        unsigned window_width = 15) noexcept {
  return make_in_memory_partition_table_accessor_impl<U, T>(generators, alloc, window_width);
}

template <class T>
std::unique_ptr<partition_table_accessor<T>>
make_in_memory_partition_table_accessor(basct::cspan<T> generators,
                                        basm::alloc_t alloc = memr::get_pinned_resource(),
                                        unsigned window_width = 15) noexcept {
  return make_in_memory_partition_table_accessor_impl<T, T>(generators, alloc, window_width);
}
} // namespace sxt::mtxpp2
