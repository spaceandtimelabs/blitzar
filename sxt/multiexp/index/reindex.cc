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
#include "sxt/multiexp/index/reindex.h"

#include <algorithm>
#include <utility>
#include <vector>

namespace sxt::mtxi {
//--------------------------------------------------------------------------------------------------
// heap_element
//--------------------------------------------------------------------------------------------------
namespace {
using heap_element = std::pair<uint64_t*, uint64_t>;
} // namespace

//--------------------------------------------------------------------------------------------------
// comp
//--------------------------------------------------------------------------------------------------
namespace {
struct comp {
  bool operator()(heap_element lhs, heap_element rhs) const noexcept {
    return *lhs.first > *rhs.first;
  }
};
} // namespace

//--------------------------------------------------------------------------------------------------
// init_heap
//--------------------------------------------------------------------------------------------------
static void init_heap(std::vector<heap_element>& heap, basct::span<basct::span<uint64_t>> rows,
                      basf::function_ref<size_t(basct::cspan<uint64_t>)> offset_functor) noexcept {
  heap.reserve(rows.size());
  for (size_t row_index = 0; row_index < rows.size(); ++row_index) {
    auto row = rows[row_index];
    if (row.empty()) {
      continue;
    }
    heap.emplace_back(row.data() + offset_functor(row), row_index);
  }
  std::make_heap(heap.begin(), heap.end(), comp{});
}

//--------------------------------------------------------------------------------------------------
// advance_heap
//--------------------------------------------------------------------------------------------------
static void advance_heap(std::vector<heap_element>& heap,
                         basct::span<basct::span<uint64_t>> rows) noexcept {
  auto& [ptr, index] = heap.back();
  ++ptr;
  auto row = rows[index];
  if (ptr == row.end()) {
    heap.resize(heap.size() - 1);
  } else {
    std::push_heap(heap.begin(), heap.end(), comp{});
  }
}

//--------------------------------------------------------------------------------------------------
// pop_first
//--------------------------------------------------------------------------------------------------
static void pop_first(uint64_t*& out, std::vector<heap_element>& heap,
                      basct::span<basct::span<uint64_t>> rows) noexcept {
  std::pop_heap(heap.begin(), heap.end(), comp{});
  *out = *heap.back().first;
  *heap.back().first = 0;
  advance_heap(heap, rows);
}

//--------------------------------------------------------------------------------------------------
// pop
//--------------------------------------------------------------------------------------------------
static void pop(uint64_t*& out, size_t& working_index, std::vector<heap_element>& heap,
                basct::span<basct::span<uint64_t>> rows) noexcept {
  std::pop_heap(heap.begin(), heap.end(), comp{});
  auto& ptr = heap.back().first;
  if (*out != *ptr) {
    *++out = *ptr;
    *ptr = ++working_index;
  } else {
    *ptr = working_index;
  }
  advance_heap(heap, rows);
}

//--------------------------------------------------------------------------------------------------
// reindex_rows
//--------------------------------------------------------------------------------------------------
void reindex_rows(basct::span<basct::span<uint64_t>> rows, basct::span<uint64_t>& values,
                  basf::function_ref<size_t(basct::cspan<uint64_t>)> offset_functor) noexcept {
  if (rows.empty()) {
    values = {};
    return;
  }
  std::vector<heap_element> heap;
  init_heap(heap, rows, offset_functor);

  auto out = values.begin();
  pop_first(out, heap, rows);
  uint64_t working_index = 0;
  while (!heap.empty()) {
    pop(out, working_index, heap, rows);
  }

  values = {values.data(), working_index + 1};
}

void reindex_rows(basct::span<basct::span<uint64_t>> rows, basct::span<uint64_t>& values) noexcept {
  auto offset_functor = [](basct::cspan<uint64_t> /*row*/) noexcept { return 0; };
  reindex_rows(rows, values, offset_functor);
}
} // namespace sxt::mtxi
