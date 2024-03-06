#pragma once

#include <cstdint>

#include "sxt/base/curve/element.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// reduce_partitions 
//--------------------------------------------------------------------------------------------------
template <unsigned ItemsPerThreadLg2, bascrv::element T>
__global__ void reduce_partitions(T items[(1u << ItemsPerThreadLg2)], const uint16_t* __restrict__ bitsets,
                                  const T* __restrict__ table, unsigned num_products) noexcept {
  constexpr unsigned items_per_thread = 1u << ItemsPerThreadLg2;
  constexpr unsigned num_entries = (1u << 16u);

  // load keys
  uint16_t keys[items_per_thread];
  for (unsigned i=0; i<items_per_thread; ++i) {
    keys[i] = bitsets[i * num_products];
  }

  // load curve elements
  for (unsigned i=0; i<items_per_thread; ++i) {
    items[i] = table[keys[i]];
    table += num_entries;
  }

  // reduce
  for (unsigned i = ItemsPerThreadLg2 - 1u; i-- > 0u;) {
    auto k = 1u << i;
    for (unsigned j = 0u; j < k; ++j) {
      add_inplace(items[j], items[k + j]);
    }
  }
}

//--------------------------------------------------------------------------------------------------
// product_kernel 
//--------------------------------------------------------------------------------------------------
template <unsigned ItemsPerThreadLg2, bascrv::element T>
__global__ void product_kernel(T* __restrict__ products, const uint16_t* __restrict__ bitsets,
                               const T* __restrict__ table, unsigned num_products,
                               unsigned n) {
  constexpr auto items_per_thread = 1u << ItemsPerThreadLg2;

  auto product_index = blockIdx.x * blockDim.x + threadIdx.x;

  // adjust pointers
  bitsets += product_index * num_products;

  // lo
  (void)products;
  (void)bitsets;
  (void)table;
  (void)num_products;
  (void)n;
}
} // namespace sxt::mtxpp2
