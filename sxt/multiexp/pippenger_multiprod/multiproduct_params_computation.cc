#include "sxt/multiexp/pippenger_multiprod/multiproduct_params_computation.h"

#include <cmath>

#include "sxt/multiexp/pippenger_multiprod/multiproduct_params.h"

namespace sxt::mtxpmp {
//--------------------------------------------------------------------------------------------------
// compute_multiproduct_params
//--------------------------------------------------------------------------------------------------
void compute_multiproduct_params(multiproduct_params& params, size_t num_outputs,
                                 size_t num_inputs) noexcept {
  if (num_inputs == 0) {
    params = {};
    return;
  }
  params.partition_size = static_cast<size_t>(std::ceil(std::log2(num_outputs * num_inputs)));
  if (params.partition_size <= 3 || params.partition_size >= num_inputs / 2) {
    params.partition_size = 0;
  }
  auto clump_size = std::ceil(num_inputs / std::log2(num_inputs + 1));
  params.input_clump_size = clump_size;
  params.output_clump_size = clump_size;
}
} // namespace sxt::mtxpmp
