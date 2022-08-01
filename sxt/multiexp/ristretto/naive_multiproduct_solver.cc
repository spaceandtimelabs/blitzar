#include "sxt/multiexp/ristretto/naive_multiproduct_solver.h"

#include <cassert>

#include "sxt/base/container/span.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/multiexp/index/index_table.h"

namespace sxt::mtxrs {
//--------------------------------------------------------------------------------------------------
// workspace_size
//--------------------------------------------------------------------------------------------------
size_t naive_multiproduct_solver::workspace_size(size_t num_multiproduct_inputs,
                                                 size_t /*num_entries*/) const noexcept {
  return num_multiproduct_inputs;
}

//--------------------------------------------------------------------------------------------------
// solve
//--------------------------------------------------------------------------------------------------
void naive_multiproduct_solver::solve(memmg::managed_array<void>& inout,
                                      mtxi::index_table& multiproduct_table,
                                      size_t num_inputs) const noexcept {
  basct::cspan<c21t::element_p3> inputs{static_cast<c21t::element_p3*>(inout.data()), inout.size()};
  memmg::managed_array<c21t::element_p3> outputs{multiproduct_table.num_rows(),
                                                 inout.get_allocator()};
  for (size_t row_index = 0; row_index < multiproduct_table.num_rows(); ++row_index) {
    auto products = multiproduct_table.header()[row_index];
    assert(products.size() > 2);
    c21t::element_p3 output = inputs[products[2]];
    for (size_t input_index = 3; input_index < products.size(); ++input_index) {
      auto input = products[input_index];
      c21o::add(output, output, inputs[input]);
    }
    outputs[row_index] = output;
  }
  inout = std::move(outputs);
}
} // namespace sxt::mtxrs
