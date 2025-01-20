#pragma once

#include <cstddef>

#include "sxt/proof/sumcheck/constant.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// random_sumcheck_descriptor
//--------------------------------------------------------------------------------------------------
struct random_sumcheck_descriptor {
  size_t min_length = 1;
  size_t max_length = 10;

  size_t min_num_products = 1;
  size_t max_num_products = 5;

  size_t min_product_length = 1;
  size_t max_product_length = max_degree_v;

  size_t min_num_mles = 1;
  size_t max_num_mles = 5;
};
} // namespace sxt::prfsk
