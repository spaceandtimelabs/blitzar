#include "sxt/multiexp/ristretto/multiexponentiation_cpu_driver.h"

#include <cassert>

#include "sxt/base/bit/count.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/base/container/span.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/type/element_p1p1.h"
#include "sxt/curve21/type/conversion_utility.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::mtxrs {
//--------------------------------------------------------------------------------------------------
// compute_multiproduct_inputs
//--------------------------------------------------------------------------------------------------
void multiexponentiation_cpu_driver::compute_multiproduct_inputs(
       memmg::managed_array<void>& inout,
       basct::cspan<basct::cspan<size_t>> powers,
       size_t radix_log2) const noexcept {
  
  assert(inout.size() == powers.size());
  size_t num_terms = 0;
  
  for (auto& subpowers : powers) {
    num_terms += subpowers.size();
  }
  
  basct::cspan<rstt::compressed_element> inputs {
    static_cast<rstt::compressed_element*>(inout.data()), inout.size()
  };
  
  size_t term_index = 0;
  memmg::managed_array<c21t::element_p3> terms{num_terms, inout.get_allocator()};
  
  // we compute the power set for each input generator, such as:
  // inout = {2^0 * a, 2^3 * a, 2^6 * a, 2^0 * b, 2^3 * b, 2^6 * b, 2^0 * c, 2^3 * c},
  // where {a, b, c} are curve255 points given as input
  for(size_t input_index = 0; input_index < inputs.size(); ++input_index) {
    size_t ith_iter = 0;
    size_t input_power = 0;

    c21t::element_p3 p;
    rstb::from_bytes(p, inputs[input_index].data());

    // we use a `while` loop to compute in one pass the terms
    // 2^0 * p, 2^1 * p, 2^2 * p, 2^3 * p, ..., 2^(powers[input_index] * radix_log2) * p
    while (input_power < powers[input_index].size()) {
      // we check if we found one of the powers that we desired, such as:
      // 2^(0 * radix_log2), 2^(1 * radix_log2), 2^(2 * radix_log2), ...
      if (ith_iter == powers[input_index][input_power] * radix_log2) {
        input_power++;
        terms[term_index++] = p;

        // we compare the current input_power with the previous, input_power - 1
        if (input_power < powers[input_index].size()) {
          // because we must guarantee that `powers[input_index]` array is sorted
          assert(powers[input_index][input_power - 1]
                    < powers[input_index][input_power]);
        }
      }

      // at each iteration, we double p so that
      // at the i-th iteration we have:  2^i * p
      c21t::element_p1p1 temp_p;
      c21o::double_element(temp_p, p);
      c21t::to_element_p3(p, temp_p);

      ith_iter++;
    }
  }
  
  inout = std::move(terms);
}

//--------------------------------------------------------------------------------------------------
// compute_multiproduct
//--------------------------------------------------------------------------------------------------
void multiexponentiation_cpu_driver::compute_multiproduct(
    memmg::managed_array<void>& inout,
    mtxi::index_table& multiproduct_table) const noexcept {

  basct::cspan<c21t::element_p3> inputs{static_cast<c21t::element_p3*>(inout.data()),
                                inout.size()};

  memmg::managed_array<c21t::element_p3> outputs{multiproduct_table.num_rows(),
                                         inout.get_allocator()};

  for (size_t row_index = 0; row_index < multiproduct_table.num_rows();
       ++row_index) {
    auto products = multiproduct_table.header()[row_index];

    assert(!products.empty());

    assert(products[0] < inputs.size());

    // we manually set the first `input_index`
    // to prevent one `add` operation
    c21t::element_p3 output = inputs[products[0]];

    for (size_t input_index = 1; input_index < products.size(); ++input_index) {
      auto input = products[input_index];
      assert(input < inputs.size());
      c21o::add(output, output, inputs[input]);
    }

    outputs[row_index] = output;
  }
  
  inout = std::move(outputs);
}

//--------------------------------------------------------------------------------------------------
// combine_multiproduct_outputs
//--------------------------------------------------------------------------------------------------
void multiexponentiation_cpu_driver::combine_multiproduct_outputs(
    memmg::managed_array<void>& inout,
    basct::cspan<uint8_t> output_digit_or_all) const noexcept {

  basct::cspan<c21t::element_p3> inputs{static_cast<c21t::element_p3*>(inout.data()),
                                inout.size()};

  memmg::managed_array<rstt::compressed_element> outputs{output_digit_or_all.size(),
                                         inout.get_allocator()};
  size_t input_index = 0;
  for (size_t output_index = 0; output_index < output_digit_or_all.size();
       ++output_index) {
    uint64_t digit_or_all = output_digit_or_all[output_index];

    int digit_count_one = basbt::pop_count(digit_or_all);

    if (digit_count_one == 0) {
      memset(outputs[output_index].data(), 0, sizeof(rstt::compressed_element));
      
      continue;
    }
    
    int digit_bit_index = 8 * sizeof(digit_or_all) - basbt::count_leading_zeros(digit_or_all) - 1;

    input_index += digit_count_one;

    c21t::element_p1p1 res2_p1p1;

    // we manually set the first `input_index`
    // to prevent one `double` and one `add` operation
    auto output = inputs[--input_index];

    // The following implementation uses the formula:
    // output = 2^{i_0} * a0 + 2^{i_1} * (a1 + 2^{i_2} * (a2 + ..)..),
    // where i_0 <= i_1 <= i_2 <= ... <= i_n.
    while (digit_bit_index-- > 0) {
      c21o::double_element(res2_p1p1, output);
      c21t::to_element_p3(output, res2_p1p1);

      // if i-th bit is set, we need to add a_i to output
      if (digit_or_all & (1ull << digit_bit_index)) {
        c21o::add(output, output, inputs[--input_index]);
      }
    }

    input_index += digit_count_one;

    rstb::to_bytes(outputs[output_index].data(), output);
  }

  inout = std::move(outputs);
}
}  // namespace sxt::mtxrs
