#pragma once

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// sum_round
//--------------------------------------------------------------------------------------------------
template <size_t MaxDegree, class Scalar>
void sum_round(Scalar* __restrict__ polynomial, const Scalar* __restrict__ mles,
               const unsigned* __restrict__ product_table,
               const unsigned* __restrict__ product_lengths, unsigned num_products) noexcept {
  (void)polynomial;
  (void)mles;
  (void)product_table;
  (void)product_lengths;
  (void)num_products;
}
} // namespace sxt::prfsk
