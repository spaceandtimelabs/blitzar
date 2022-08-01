#include "sxt/multiexp/ristretto/multiexponentiation_cpu_driver.h"

#include <cassert>

#include "sxt/base/bit/count.h"
#include "sxt/base/container/span.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/type/conversion_utility.h"
#include "sxt/curve21/type/element_p1p1.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/ristretto/compressed_input_accessor.h"
#include "sxt/multiexp/ristretto/naive_multiproduct_solver.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::mtxrs {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
multiexponentiation_cpu_driver::multiexponentiation_cpu_driver(
    const mtxrs::input_accessor* input_accessor,
    const mtxrs::multiproduct_solver* multiproduct_solver) noexcept
    : input_accessor_{input_accessor}, multiproduct_solver_{multiproduct_solver} {}

//--------------------------------------------------------------------------------------------------
// compute_multiproduct_inputs
//--------------------------------------------------------------------------------------------------
void multiexponentiation_cpu_driver::compute_multiproduct_inputs(
    memmg::managed_array<void>& inout, basct::cspan<basct::cspan<size_t>> powers, size_t radix_log2,
    size_t num_multiproduct_inputs, size_t num_multiproduct_entries) const noexcept {
  compressed_input_accessor default_accessor;
  auto accessor = input_accessor_;
  if (accessor == nullptr) {
    accessor = &default_accessor;
  }
  auto num_inputs = powers.size();
  auto inputs = inout.data();

  size_t term_index = 0;
  size_t multiproduct_workspace_size =
      multiproduct_solver_ == nullptr
          ? num_multiproduct_inputs
          : multiproduct_solver_->workspace_size(num_multiproduct_inputs, num_multiproduct_entries);
  memmg::managed_array<c21t::element_p3> terms{multiproduct_workspace_size, inout.get_allocator()};

  // we compute the power set for each input generator, such as:
  // inout = {2^0 * a, 2^3 * a, 2^6 * a, 2^0 * b, 2^3 * b, 2^6 * b, 2^0 * c, 2^3 * c},
  // where {a, b, c} are curve255 points given as input
  for (size_t input_index = 0; input_index < num_inputs; ++input_index) {
    size_t ith_iter = 0;
    size_t input_power = 0;

    c21t::element_p3 p;
    accessor->get_element(p, inputs, input_index);

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
          assert(powers[input_index][input_power - 1] < powers[input_index][input_power]);
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
void multiexponentiation_cpu_driver::compute_multiproduct(memmg::managed_array<void>& inout,
                                                          mtxi::index_table& multiproduct_table,
                                                          size_t num_inputs) const noexcept {
  naive_multiproduct_solver naive_solver;
  auto solver = multiproduct_solver_ == nullptr ? &naive_solver : multiproduct_solver_;
  solver->solve(inout, multiproduct_table, num_inputs);
}

//--------------------------------------------------------------------------------------------------
// combine_multiproduct_outputs
//--------------------------------------------------------------------------------------------------
void multiexponentiation_cpu_driver::combine_multiproduct_outputs(
    memmg::managed_array<void>& inout, basct::cspan<uint8_t> output_digit_or_all) const noexcept {

  basct::cspan<c21t::element_p3> inputs{static_cast<c21t::element_p3*>(inout.data()), inout.size()};

  memmg::managed_array<rstt::compressed_element> outputs{output_digit_or_all.size(),
                                                         inout.get_allocator()};
  size_t input_index = 0;
  for (size_t output_index = 0; output_index < output_digit_or_all.size(); ++output_index) {
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
} // namespace sxt::mtxrs
