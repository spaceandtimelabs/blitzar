#pragma once

#include <cstddef>

namespace sxt::mtxrn {
//--------------------------------------------------------------------------------------------------
// random_multiproduct_descriptor
//--------------------------------------------------------------------------------------------------
struct random_multiproduct_descriptor {
  size_t min_sequence_length = 1;
  size_t max_sequence_length = 10;
  size_t min_num_sequences = 2;
  size_t max_num_sequences = 10;
  size_t max_num_inputs = 10;
};
}  // namespace sxt::mtxrn
