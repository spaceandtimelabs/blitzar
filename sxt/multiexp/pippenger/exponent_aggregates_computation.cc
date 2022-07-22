#include "sxt/multiexp/pippenger/exponent_aggregates_computation.h"

#include <algorithm>

#include "sxt/base/bit/count.h"
#include "sxt/multiexp/base/exponent.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/base/exponent_utility.h"
#include "sxt/multiexp/pippenger/exponent_aggregates.h"

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// compute_exponent_aggregates
//--------------------------------------------------------------------------------------------------
void compute_exponent_aggregates(exponent_aggregates& aggregates,
                                 basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  size_t max_sequence_length = 0;
  for (auto& sequence : exponents) {
    max_sequence_length = std::max(max_sequence_length, sequence.n);
  }
  aggregates.term_or_all.clear();
  aggregates.term_or_all.resize(max_sequence_length);
  aggregates.output_or_all.clear();
  aggregates.output_or_all.resize(exponents.size());
  aggregates.pop_count = 0;

  mtxb::exponent max_exponent;
  for (size_t output_index = 0; output_index < exponents.size(); ++output_index) {
    auto& sequence = exponents[output_index];
    auto element_nbytes = sequence.element_nbytes;
    for (size_t term_index = 0; term_index < sequence.n; ++term_index) {
      auto term_data = sequence.data + term_index * element_nbytes;
      mtxb::or_equal(aggregates.term_or_all[term_index], {term_data, element_nbytes});
      mtxb::or_equal(aggregates.output_or_all[output_index], {term_data, element_nbytes});
      mtxb::exponent e;
      std::copy_n(term_data, element_nbytes, e.data());
      aggregates.pop_count += basbt::pop_count(term_data, element_nbytes);
      if (max_exponent < e) {
        max_exponent = e;
      }
    }
  }
  aggregates.max_exponent = max_exponent;
}
} // namespace sxt::mtxpi
