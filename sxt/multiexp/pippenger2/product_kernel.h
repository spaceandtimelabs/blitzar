#pragma once

#include <cstdint>

#include "sxt/base/curve/element.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// reduce_partitions 
//--------------------------------------------------------------------------------------------------
template <unsigned ItemsPerThreadLg2, bascrv::element T>
__device__ void reduce_partitions(T items[(1u << ItemsPerThreadLg2)], const uint16_t* __restrict__ bitsets,
                                  const T* __restrict__ table, unsigned num_products) noexcept {
  constexpr unsigned items_per_thread = 1u << ItemsPerThreadLg2;
  constexpr unsigned num_entries = (1u << 16u);

  // load keys
  uint16_t keys[items_per_thread];
  for (unsigned i=0; i<items_per_thread; ++i) {
    keys[i] = bitsets[i * num_products];
  }

  // load curve elements
  for (unsigned i = 0; i < items_per_thread; ++i) {
    items[i] = table[keys[i] + i * num_entries];
  }

  // reduce
  for (int i = ItemsPerThreadLg2; i-- > 0;) {
    auto k = 1u << i;
    for (int j = 0; j < k; ++j) {
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
  constexpr unsigned num_entries = (1u << 16u);

  auto product_index = blockIdx.x * blockDim.x + threadIdx.x;
  if (product_index >= num_products) {
    return;
  }

  // adjust pointers
  bitsets += product_index;
  products += product_index;

  // compute first product
  T items[items_per_thread];
  reduce_partitions<ItemsPerThreadLg2>(items, bitsets, table, num_products);

  // reduce rest
  T res = items[0];
  for (unsigned i=items_per_thread; i<n; i+=items_per_thread) {
    bitsets += items_per_thread * num_products;
    table += items_per_thread * num_entries;
    reduce_partitions<ItemsPerThreadLg2>(items, bitsets, table, num_products);
    add_inplace(res, items[0]);
  }

  // write result
  *products = res;
}
} // namespace sxt::mtxpp2
