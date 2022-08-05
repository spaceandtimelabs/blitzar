#pragma once

#include <cassert>
#include <cstddef>

#include "sxt/multiexp/bitset_multiprod/value_cache.h"

namespace sxt::mtxbmp {
//--------------------------------------------------------------------------------------------------
// compute_cache_size
//--------------------------------------------------------------------------------------------------
size_t compute_cache_size(size_t num_terms) noexcept;

//--------------------------------------------------------------------------------------------------
// init_value_cache_side
//--------------------------------------------------------------------------------------------------
template <class T, class Op>
void init_value_cache_side(basct::span<T> cache_side, Op op, basct::cspan<T> terms) noexcept {
  for (auto& value : cache_side) {
    op.mark_unset(value);
  }
  for (size_t term_index = 0; term_index < terms.size(); ++term_index) {
    cache_side[(1 << term_index) - 1] = terms[term_index];
  }
}

//--------------------------------------------------------------------------------------------------
// init_value_cache
//--------------------------------------------------------------------------------------------------
template <class T, class Op>
void init_value_cache(value_cache<T> cache, Op op, basct::cspan<T> terms) noexcept {
  assert(terms.size() == cache.num_terms());
  auto [left_cache, right_cache] = cache.split();
  auto left_num_terms = cache.half_num_terms();
  init_value_cache_side(left_cache, op, terms.subspan(0, left_num_terms));
  init_value_cache_side(right_cache, op, terms.subspan(left_num_terms));
}
} // namespace sxt::mtxbmp
