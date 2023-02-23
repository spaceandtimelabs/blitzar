#include "sxt/multiexp/curve21/multiexponentiation_gpu_driver.h"

#include "sxt/base/container/blob_array.h"
#include "sxt/base/container/span.h"
#include "sxt/base/container/span_void.h"
#include "sxt/curve21/operation/accumulator.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/curve21/multiproducts_combination.h"
#include "sxt/multiexp/index/index_table.h"
#include "sxt/multiexp/multiproduct_gpu/multiproduct.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// compute_multiproduct
//--------------------------------------------------------------------------------------------------
xena::future<memmg::managed_array<void>> multiexponentiation_gpu_driver::compute_multiproduct(
    mtxi::index_table&& multiproduct_table, basct::span_cvoid generators,
    const basct::blob_array& masks, size_t num_inputs) const noexcept {
  return mtxmpg::compute_multiproduct<c21o::accumulator>(
      {static_cast<const c21t::element_p3*>(generators.data()), generators.size()},
      multiproduct_table.cheader(), masks, num_inputs);
}

//--------------------------------------------------------------------------------------------------
// combine_multiproduct_outputs
//--------------------------------------------------------------------------------------------------
xena::future<memmg::managed_array<void>>
multiexponentiation_gpu_driver::combine_multiproduct_outputs(
    xena::future<memmg::managed_array<void>>&& multiproduct,
    basct::blob_array&& output_digit_or_all) const noexcept {
  return multiproduct.then(
      [output_digit_or_all = std::move(output_digit_or_all)](
          memmg::managed_array<void>&& products) noexcept -> memmg::managed_array<void> {
        memmg::managed_array<c21t::element_p3> res(output_digit_or_all.size());
        combine_multiproducts(res, output_digit_or_all, products.as_array<c21t::element_p3>());
        return res;
      });
}
} // namespace sxt::mtxc21
