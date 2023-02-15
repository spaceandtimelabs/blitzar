#include "sxt/multiexp/curve21/multiexponentiation_cpu_driver.h"

#include "sxt/base/bit/span_op.h"
#include "sxt/base/container/blob_array.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/curve21/doubling_reduction.h"
#include "sxt/multiexp/curve21/multiproduct_solver.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// constructor
//--------------------------------------------------------------------------------------------------
multiexponentiation_cpu_driver::multiexponentiation_cpu_driver(
    const multiproduct_solver* solver) noexcept
    : solver_{solver} {}

//--------------------------------------------------------------------------------------------------
// compute_multiproduct
//--------------------------------------------------------------------------------------------------
xena::future<memmg::managed_array<void>> multiexponentiation_cpu_driver::compute_multiproduct(
    mtxi::index_table&& multiproduct_table, basct::span_cvoid generators,
    const basct::blob_array& masks, size_t num_inputs) const noexcept {
  auto res =
      solver_
          ->solve(std::move(multiproduct_table),
                  {static_cast<const c21t::element_p3*>(generators.data()), generators.size()},
                  masks, num_inputs)
          .await_result();
  return xena::make_ready_future<memmg::managed_array<void>>(std::move(res));
}

//--------------------------------------------------------------------------------------------------
// combine_multiproduct_outputs
//--------------------------------------------------------------------------------------------------
xena::future<memmg::managed_array<void>>
multiexponentiation_cpu_driver::combine_multiproduct_outputs(
    xena::future<memmg::managed_array<void>>&& multiproduct,
    basct::blob_array&& output_digit_or_all) const noexcept {
  auto products = multiproduct.await_result().as_array<c21t::element_p3>();
  memmg::managed_array<c21t::element_p3> res(output_digit_or_all.size());
  size_t input_index = 0;
  for (size_t output_index = 0; output_index < output_digit_or_all.size(); ++output_index) {
    auto digit_or_all = output_digit_or_all[output_index];
    int digit_count_one = basbt::pop_count(digit_or_all);
    if (digit_count_one == 0) {
      res[output_index] = c21cn::zero_p3_v;
      continue;
    }
    c21t::element_p3 output;
    SXT_DEBUG_ASSERT(input_index + digit_count_one <= products.size());
    doubling_reduce(output, digit_or_all,
                    basct::cspan<c21t::element_p3>{&products[input_index],
                                                   static_cast<size_t>(digit_count_one)});
    input_index += digit_count_one;
    res[output_index] = output;
  }
  return xena::make_ready_future<memmg::managed_array<void>>(std::move(res));
}
} // namespace sxt::mtxc21
