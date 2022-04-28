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
static void init_heap(std::vector<heap_element>& heap,
                      basct::span<basct::span<uint64_t>> rows) noexcept {
  heap.reserve(rows.size());
  for (size_t index=0; index<rows.size(); ++index) {
    auto row = rows[index];
    if (row.empty()) {
      continue;
    }
    heap.emplace_back(row.data(), index);
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
void reindex_rows(basct::span<basct::span<uint64_t>> rows,
                  basct::span<uint64_t>& values) noexcept {
  if (rows.empty()) {
    values = {};
    return;
  }
  std::vector<heap_element> heap;
  init_heap(heap, rows);

  auto out = values.begin();
  pop_first(out, heap, rows);
  uint64_t working_index = 0;
  while(!heap.empty()) {
    pop(out, working_index, heap, rows);
  }

  values = {values.data(), working_index + 1};
}
}  // namespace sxt::mtxi
