#include "sxt/multiexp/curve21/naive_multiproduct_solver.h"

#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/base/generator_utility.h"
#include "sxt/multiexp/index/index_table.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// solve
//--------------------------------------------------------------------------------------------------
xena::future<memmg::managed_array<c21t::element_p3>>
naive_multiproduct_solver::solve(mtxi::index_table&& multiproduct_table,
                                 basct::cspan<c21t::element_p3> generators,
                                 const basct::blob_array& masks, size_t num_inputs) const noexcept {
  memmg::managed_array<c21t::element_p3> inputs_data;
  basct::cspan<c21t::element_p3> inputs;
  if (num_inputs == generators.size()) {
    inputs = generators;
  } else {
    inputs_data = memmg::managed_array<c21t::element_p3>(num_inputs);
    mtxb::filter_generators<c21t::element_p3>(inputs_data, generators, masks);
    inputs = inputs_data;
  }
  memmg::managed_array<c21t::element_p3> res(multiproduct_table.num_rows());
  for (size_t row_index = 0; row_index < multiproduct_table.num_rows(); ++row_index) {
    auto products = multiproduct_table.header()[row_index];
    SXT_DEBUG_ASSERT(products.size() > 2);
    c21t::element_p3 output = inputs[products[2]];
    for (size_t input_index = 3; input_index < products.size(); ++input_index) {
      auto input = products[input_index];
      c21o::add(output, output, inputs[input]);
    }
    res[row_index] = output;
  }
  return xena::make_ready_future(std::move(res));
}
} // namespace sxt::mtxc21
