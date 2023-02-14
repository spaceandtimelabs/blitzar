#pragma once

#include <cstddef>

namespace sxt::mtxrn {
//--------------------------------------------------------------------------------------------------
// random_multiexponentiation_descriptor
//--------------------------------------------------------------------------------------------------
struct random_multiexponentiation_descriptor {
  size_t min_sequence_length = 1;
  size_t max_sequence_length = 10;
  size_t min_exponent_num_bytes = 1;
  size_t max_exponent_num_bytes = 32;
};

struct random_multiexponentiation_descriptor2 {
  size_t min_num_sequences = 1;
  size_t max_num_sequences = 1;
  size_t min_sequence_length = 1;
  size_t max_sequence_length = 10;
  size_t min_exponent_num_bytes = 1;
  size_t max_exponent_num_bytes = 32;
};
} // namespace sxt::mtxrn
