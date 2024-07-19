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

#include <memory>
#include <string_view>

#include "sxt/base/container/span.h"
#include "sxt/base/type/raw_stream.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor_base.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// partition_table_accessor
//--------------------------------------------------------------------------------------------------
/**
 * Support accessing precomputed sums for groups of 16 generators.
 *
 * For example, if there are 32 generators
 *
 *    g0, ..., g15, g16, ..., g31
 *
 * an accessor will contain two tables each of 2^16 entries with all the sums of
 * generators g0 to g15 and all the sums of generators g16 to g31, respectively.
 */
template <class T> class partition_table_accessor : public partition_table_accessor_base {
public:
  virtual unsigned window_width() const noexcept { return 16u; }

  /**
   * Asynchronously copy precomputed sums of partitions to device.
   *
   * `first` specifies the partition group offset to use.
   */
  virtual void async_copy_to_device(basct::span<T> dest, bast::raw_stream_t stream,
                                    unsigned first) const noexcept = 0;

  /**
   * Make a view into precomputed sums of partitions available to host memory.
   *
   * `first` specifies the partition group offset to use. If memory needs to be allocated
   * to make the view available, it will be allocated using alloc. Make sure that alloc uses
   * a resource that frees memory upon destruction (e.g. std::pmr::monotonic_buffer_resource).
   */
  virtual basct::cspan<T> host_view(std::pmr::polymorphic_allocator<> alloc, unsigned first,
                                    unsigned size) const noexcept = 0;

  virtual void write_to_file(std::string_view filename) const noexcept = 0;
};
} // namespace sxt::mtxpp2
