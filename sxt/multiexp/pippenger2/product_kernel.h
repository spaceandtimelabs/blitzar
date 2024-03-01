#pragma once

#include <cstdint>

#include "sxt/base/curve/element.h"

namespace sxt::mtxpp2 {
//--------------------------------------------------------------------------------------------------
// product_kernel 
//--------------------------------------------------------------------------------------------------
template <bascrv::element T>
__global__ void product_kernel(T* __restrict__ products, const uint16_t* __restrict__ bitsets,
                               const T* __restrict__ table, unsigned num_products,
                               unsigned n) {
  (void)products;
  (void)bitsets;
  (void)table;
  (void)num_products;
  (void)n;
}
} // namespace sxt::mtxpp2
