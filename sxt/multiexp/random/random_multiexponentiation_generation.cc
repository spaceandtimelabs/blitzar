#include "sxt/multiexp/random/random_multiexponentiation_generation.h"

#include "sxt/base/error/assert.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/random/random_multiexponentiation_descriptor.h"

namespace sxt::mtxrn {
//--------------------------------------------------------------------------------------------------
// generate_random_multiexponentiation
//--------------------------------------------------------------------------------------------------
void generate_random_multiexponentiation(
    uint64_t& num_inputs, basct::span<mtxb::exponent_sequence> exponents, basm::alloc_t alloc,
    std::mt19937& rng, const random_multiexponentiation_descriptor& descriptor) noexcept {

  num_inputs = 0;

  for (size_t curr_sequence = 0; curr_sequence < exponents.size(); ++curr_sequence) {
    std::uniform_int_distribution<uint8_t> data_gen(0, 255);
    std::uniform_int_distribution<size_t> sequence_gen(descriptor.min_sequence_length,
                                                       descriptor.max_sequence_length);
    std::uniform_int_distribution<uint8_t> exponent_gen(descriptor.min_exponent_num_bytes,
                                                        descriptor.max_exponent_num_bytes);

    SXT_DEBUG_ASSERT(descriptor.min_exponent_num_bytes > 0);
    SXT_DEBUG_ASSERT(descriptor.min_sequence_length <= descriptor.max_sequence_length);
    SXT_DEBUG_ASSERT(descriptor.min_exponent_num_bytes <= descriptor.max_exponent_num_bytes);

    size_t sequence_length = sequence_gen(rng);
    uint8_t exponent_length = exponent_gen(rng);
    size_t sequence_total_num_bytes = sequence_length * exponent_length;

    num_inputs = std::max(num_inputs, sequence_length);

    auto exponent_array = alloc.allocate(sequence_total_num_bytes);

    for (size_t j = 0; j < sequence_total_num_bytes; ++j) {
      exponent_array[j] = static_cast<std::byte>(data_gen(rng));
    }

    exponents[curr_sequence] = {.element_nbytes = exponent_length,
                                .n = sequence_length,
                                .data = reinterpret_cast<const uint8_t*>(exponent_array)};
  }
}

} // namespace sxt::mtxrn
