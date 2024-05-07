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

#include <string_view>
#include <memory>

#include "sxt/base/container/span.h"
#include "sxt/base/curve/element.h"
#include "sxt/base/type/raw_stream.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// partition_table_accessor
//--------------------------------------------------------------------------------------------------
template <bascrv::element T> class partition_table_accessor {
public:
  virtual ~partition_table_accessor() noexcept = default;

  virtual void async_copy_precomputed_sums_to_device(basct::span<T> dest, bast::raw_stream_t stream,
                                                     unsigned first) const noexcept = 0;

  virtual basct::cspan<T> host_view(std::pmr::polymorphic_allocator<> alloc, unsigned first,
                                    unsigned n) const noexcept = 0;

  virtual void write_to_file(std::string_view filename) const noexcept = 0;
};
} // namespace sxt::mtxpp2
