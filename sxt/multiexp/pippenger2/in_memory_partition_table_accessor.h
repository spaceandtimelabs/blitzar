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

#include <cerrno>
#include <cstring>
#include <fstream>
#include <string_view>

#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/error/panic.h"
#include "sxt/base/memory/alloc.h"
#include "sxt/base/num/divide_up.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/pippenger2/partition_table_accessor.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// in_memory_partition_table_accessor
//--------------------------------------------------------------------------------------------------
template <class T>
class in_memory_partition_table_accessor final : public partition_table_accessor<T> {

public:
  explicit in_memory_partition_table_accessor(
      std::string_view filename, basm::alloc_t alloc = memr::get_pinned_resource()) noexcept
      : table_{alloc} {
    std::ifstream in{std::string{filename}, std::ios::binary};
    if (!in.good()) {
      baser::panic("failed to open {}: {}", filename, std::strerror(errno));
    }
    auto pos = in.tellg();
    in.seekg(0, std::ios::end);
    auto size = in.tellg() - pos;
    in.seekg(pos);
    in.read(reinterpret_cast<char*>(&window_width_), sizeof(unsigned));
    size -= sizeof(unsigned);
    SXT_RELEASE_ASSERT(size % sizeof(T) == 0);
    table_.resize(size / sizeof(T));
    in.read(reinterpret_cast<char*>(table_.data()), size);
    partition_table_size_ = 1u << window_width_;
  }

  explicit in_memory_partition_table_accessor(memmg::managed_array<T>&& table,
                                              unsigned window_width) noexcept
      : window_width_{window_width}, partition_table_size_{1u << window_width},
        table_{std::move(table)} {}

  // partition_table_accessor
  unsigned window_width() const noexcept override { return window_width_; }

  void copy_generators(basct::span<T> generators) const noexcept override {
    auto num_groups = basn::divide_up<size_t>(generators.size(), window_width_);
    SXT_RELEASE_ASSERT(num_groups * partition_table_size_ <= table_.size());
    size_t out = 0;
    for (size_t group_index = 0; group_index<num_groups; ++group_index) {
      for (size_t j=0; j<window_width_; ++j) {
        if (out == generators.size()) {
          return;
        }
        auto offset = 1u << j;
        generators[out++] = table_[group_index * partition_table_size_ + offset];
      }
    }
  }

  void async_copy_to_device(basct::span<T> dest, bast::raw_stream_t stream,
                            unsigned first) const noexcept override {
    SXT_RELEASE_ASSERT(table_.size() >= dest.size() + first * partition_table_size_);
    SXT_DEBUG_ASSERT(basdv::is_active_device_pointer(dest.data()));
    basdv::async_copy_host_to_device(
        dest, basct::subspan(table_, first * partition_table_size_, dest.size()), stream);
  }

  basct::cspan<T> host_view(std::pmr::polymorphic_allocator<> /*alloc*/, unsigned first,
                            unsigned size) const noexcept override {
    SXT_RELEASE_ASSERT(table_.size() >= size + first * partition_table_size_);
    return basct::subspan(table_, first * partition_table_size_, size);
  }

  void write_to_file(std::string_view filename) const noexcept override {
    std::ofstream out{std::string{filename}, std::ios::binary};
    if (!out.good()) {
      baser::panic("failed to open {}: {}", filename, std::strerror(errno));
    }
    out.write(reinterpret_cast<const char*>(&window_width_), sizeof(unsigned));
    out.write(reinterpret_cast<const char*>(table_.data()), table_.size() * sizeof(T));
  }

private:
  unsigned window_width_;
  unsigned partition_table_size_;
  memmg::managed_array<T> table_;
};
} // namespace sxt::mtxpp2
