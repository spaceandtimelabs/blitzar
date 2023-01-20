#include "sxt/multiexp/ristretto/multiexponentiation_cpu_driver.h"

#include <type_traits>

#include "sxt/base/bit/count.h"
#include "sxt/base/bit/span_op.h"
#include "sxt/base/container/blob_array.h"
#include "sxt/base/container/span.h"
#include "sxt/base/error/assert.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/operation/double.h"
#include "sxt/curve21/type/conversion_utility.h"
#include "sxt/curve21/type/element_p1p1.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/ristretto/compressed_input_accessor.h"
#include "sxt/multiexp/ristretto/doubling_reduction.h"
#include "sxt/multiexp/ristretto/naive_multiproduct_solver.h"
#include "sxt/multiexp/ristretto/uncompressed_input_accessor.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::mtxrs {
//--------------------------------------------------------------------------------------------------
// combine_multiproduct_outputs_impl
//--------------------------------------------------------------------------------------------------
template <class T>
void combine_multiproduct_outputs_impl(basct::span<T> outputs,
                                       basct::cspan<c21t::element_p3> inputs,
                                       const basct::blob_array& output_digit_or_all) noexcept {
  size_t input_index = 0;
  for (size_t output_index = 0; output_index < output_digit_or_all.size(); ++output_index) {
    auto digit_or_all = output_digit_or_all[output_index];
    int digit_count_one = basbt::pop_count(digit_or_all);
    if (digit_count_one == 0) {
      std::memset(static_cast<void*>(&outputs[output_index]), 0, sizeof(T));
      continue;
    }
    c21t::element_p3 output;
    doubling_reduce(output, digit_or_all, inputs.subspan(input_index, digit_count_one));
    input_index += digit_count_one;
    if constexpr (std::is_same_v<T, rstt::compressed_element>) {
      rstb::to_bytes(outputs[output_index].data(), output);
    } else {
      outputs[output_index] = output;
    }
  }
}

//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
multiexponentiation_cpu_driver::multiexponentiation_cpu_driver(
    const mtxrs::input_accessor* input_accessor,
    const mtxrs::multiproduct_solver* multiproduct_solver, bool compress) noexcept
    : input_accessor_{input_accessor}, multiproduct_solver_{multiproduct_solver}, compress_{
                                                                                      compress} {}

//--------------------------------------------------------------------------------------------------
// compute_multiproduct_inputs
//--------------------------------------------------------------------------------------------------
void multiexponentiation_cpu_driver::compute_multiproduct_inputs(
    memmg::managed_array<void>& inout, basct::cspan<basct::cspan<size_t>> powers, size_t radix_log2,
    size_t num_multiproduct_inputs, size_t num_multiproduct_entries) const noexcept {
  compressed_input_accessor default_compressed_accessor;
  uncompressed_input_accessor default_uncompressed_accessor;
  auto accessor = input_accessor_;
  if (accessor == nullptr) {
    if (compress_) {
      accessor = &default_compressed_accessor;
    } else {
      accessor = &default_uncompressed_accessor;
    }
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
          SXT_DEBUG_ASSERT(powers[input_index][input_power - 1] < powers[input_index][input_power]);
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
    memmg::managed_array<void>& inout,
    const basct::blob_array& output_digit_or_all) const noexcept {
  basct::cspan<c21t::element_p3> inputs{static_cast<c21t::element_p3*>(inout.data()), inout.size()};
  if (compress_) {
    memmg::managed_array<rstt::compressed_element> outputs{output_digit_or_all.size(),
                                                           inout.get_allocator()};
    combine_multiproduct_outputs_impl<rstt::compressed_element>(outputs, inputs,
                                                                output_digit_or_all);
    inout = std::move(outputs);
  } else {
    memmg::managed_array<c21t::element_p3> outputs{output_digit_or_all.size(),
                                                   inout.get_allocator()};
    combine_multiproduct_outputs_impl<c21t::element_p3>(outputs, inputs, output_digit_or_all);
    inout = std::move(outputs);
  }
}
} // namespace sxt::mtxrs
