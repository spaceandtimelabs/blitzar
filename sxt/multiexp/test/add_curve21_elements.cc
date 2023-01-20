#include "sxt/multiexp/test/add_curve21_elements.h"

#include <cstdlib>
#include <iostream>

#include "sxt/base/error/panic.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/operation/add.h"
#include "sxt/curve21/type/element_p3.h"

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// add_ints
//--------------------------------------------------------------------------------------------------
void add_curve21_elements(basct::span<c21t::element_p3> result,
                          basct::cspan<basct::cspan<uint64_t>> terms,
                          basct::cspan<c21t::element_p3> inputs) noexcept {
  if (result.size() != terms.size()) {
    baser::panic("result.size() != terms.size()");
  }

  for (size_t result_index = 0; result_index < result.size(); ++result_index) {
    auto& res_i = result[result_index];
    res_i = c21cn::zero_p3_v;
    for (auto term_index : terms[result_index]) {
      if (term_index >= inputs.size()) {
        baser::panic("term_index >= inputs.size()");
      }

      c21o::add(res_i, res_i, inputs[term_index]);
    }
  }
}
} // namespace sxt::mtxtst
