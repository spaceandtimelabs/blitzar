#include "sxt/multiexp/curve21/multiexponentiation.h"

#include "sxt/base/container/blob_array.h"
#include "sxt/curve21/operation/accumulator.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/pinned_resource.h"
#include "sxt/multiexp/curve21/multiexponentiation_cpu_driver.h"
#include "sxt/multiexp/curve21/multiproducts_combination.h"
#include "sxt/multiexp/curve21/pippenger_multiproduct_solver.h"
#include "sxt/multiexp/multiproduct_gpu/multiproduct.h"
#include "sxt/multiexp/pippenger/multiexponentiation.h"
#include "sxt/multiexp/pippenger/multiproduct_decomposition.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// compute_multiexponentiation
//--------------------------------------------------------------------------------------------------
memmg::managed_array<c21t::element_p3>
compute_multiexponentiation(basct::cspan<c21t::element_p3> generators,
                            basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  pippenger_multiproduct_solver solver;
  multiexponentiation_cpu_driver driver{&solver};
  // Note: the cpu driver is non-blocking so that the future upon return the future is
  // available
  return mtxpi::compute_multiexponentiation(driver,
                                            {static_cast<const void*>(generators.data()),
                                             generators.size(), sizeof(c21t::element_p3)},
                                            exponents)
      .value()
      .as_array<c21t::element_p3>();
}

//--------------------------------------------------------------------------------------------------
// async_compute_multiexponentiation
//--------------------------------------------------------------------------------------------------
xena::future<memmg::managed_array<c21t::element_p3>>
async_compute_multiexponentiation(basct::cspan<c21t::element_p3> generators,
                                  basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  memmg::managed_array<unsigned> indexes{memr::get_pinned_resource()};
  memmg::managed_array<unsigned> product_sizes;
  basct::blob_array output_digit_or_all;
  mtxpi::compute_multiproduct_decomposition(indexes, product_sizes, output_digit_or_all, exponents);
  return mtxmpg::compute_multiproduct<c21o::accumulator>(generators, std::move(indexes),
                                                         product_sizes)
      .then([output_digit_or_all = std::move(output_digit_or_all)](
                memmg::managed_array<c21t::element_p3>&& products) noexcept {
        memmg::managed_array<c21t::element_p3> res(output_digit_or_all.size());
        combine_multiproducts(res, output_digit_or_all, products);
        return res;
      });
}
} // namespace sxt::mtxc21
