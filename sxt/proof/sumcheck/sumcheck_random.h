#pragma once

#include <vector>
#include <utility>

#include "sxt/proof/sumcheck/constant.h"

namespace sxt::s25t {
class element;
}
namespace sxt::basn {
class fast_random_number_generator;
}

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// random_sumcheck_descriptor
//--------------------------------------------------------------------------------------------------
struct random_sumcheck_descriptor {
  unsigned min_length = 1;
  unsigned max_length = 10;

  unsigned min_num_products = 1;
  unsigned max_num_products = 5;

  unsigned min_product_length = 1;
  unsigned max_product_length = max_degree_v;

  unsigned min_num_mles = 1;
  unsigned max_num_mles = 5;
};

//--------------------------------------------------------------------------------------------------
// generate_random_sumcheck_problem 
//--------------------------------------------------------------------------------------------------
void generate_random_sumcheck_problem(
    std::vector<s25t::element>& mles,
    std::vector<std::pair<s25t::element, unsigned>>& product_table,
    std::vector<unsigned>& product_terms, unsigned& n,
    basn::fast_random_number_generator& rng,
    const random_sumcheck_descriptor& descriptor) noexcept;
} // namespace sxt::prfsk
