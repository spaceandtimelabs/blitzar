#include "sxt/multiexp/test/add_ints.h"

#include <cstdlib>
#include <iostream>

namespace sxt::mtxtst {
//--------------------------------------------------------------------------------------------------
// add_ints
//--------------------------------------------------------------------------------------------------
void add_ints(basct::span<uint64_t> result, basct::cspan<basct::cspan<uint64_t>> terms,
              basct::cspan<uint64_t> inputs) noexcept {
  if (result.size() != terms.size()) {
    std::cerr << "result.size() != terms.size()\n";
    std::abort();
  }
  for (size_t result_index = 0; result_index < result.size(); ++result_index) {
    auto& res_i = result[result_index];
    res_i = 0;
    for (auto term_index : terms[result_index]) {
      if (term_index >= inputs.size()) {
        std::cerr << "term_index >= inputs.size()\n";
        std::abort();
      }
      res_i += inputs[term_index];
    }
  }
}
} // namespace sxt::mtxtst
