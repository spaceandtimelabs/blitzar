#pragma once

namespace sxt::cbnb {
//--------------------------------------------------------------------------------------------------
// sumcheck_descriptor
//--------------------------------------------------------------------------------------------------
struct sumcheck_descriptor {
  const void* mles;
  const void* product_table;
  const unsigned* product_terms;
  unsigned n;
  unsigned num_mles;
  unsigned num_products;
  unsigned num_product_terms;
  unsigned round_degree;
};
} // namespace sxt::cbnb
