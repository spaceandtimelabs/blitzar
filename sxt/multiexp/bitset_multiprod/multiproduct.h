#pragma once

#include <cassert>
#include <cstdint>

#include "sxt/base/bit/iteration.h"
#include "sxt/base/container/span.h"
#include "sxt/multiexp/bitset_multiprod/value_cache.h"

namespace sxt::mtxbmp {
//--------------------------------------------------------------------------------------------------
// lookup_or_compute
//--------------------------------------------------------------------------------------------------
template <class T, class Op>
const T& lookup_or_compute(basct::span<T> cache_side, Op op, uint64_t bitset) noexcept {
  auto& value = cache_side[bitset - 1];
  if (op.is_set(value)) {
    return value;
  }
  auto index = basbt::consume_next_bit(bitset);
  op.add(value, cache_side[(1ull << index) - 1], lookup_or_compute(cache_side, op, bitset));
  return value;
}

//--------------------------------------------------------------------------------------------------
// compute_multiproduct
//--------------------------------------------------------------------------------------------------
template <class T, class Op>
void compute_multiproduct(T& res, value_cache<T> cache, Op op, uint64_t bitset) noexcept {
  assert(bitset != 0);
  auto half_num_terms = cache.half_num_terms();
  auto right_bitset = bitset >> half_num_terms;
  auto left_bitset = bitset ^ (right_bitset << half_num_terms);
  auto [left_cache, right_cache] = cache.split();
  if (left_bitset == 0) {
    res = lookup_or_compute(right_cache, op, right_bitset);
  } else if (right_bitset == 0) {
    res = lookup_or_compute(left_cache, op, left_bitset);
  } else {
    op.add(res, lookup_or_compute(left_cache, op, left_bitset),
           lookup_or_compute(right_cache, op, right_bitset));
  }
}
} // namespace sxt::mtxbmp
