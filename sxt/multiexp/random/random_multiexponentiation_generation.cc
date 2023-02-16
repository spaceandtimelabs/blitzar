#include "sxt/multiexp/random/random_multiexponentiation_generation.h"

#include <algorithm>

#include "sxt/base/error/assert.h"
#include "sxt/base/memory/alloc_utility.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/random/int_generation.h"
#include "sxt/multiexp/random/random_multiexponentiation_descriptor.h"
#include "sxt/ristretto/random/element.h"

namespace sxt::mtxrn {
//--------------------------------------------------------------------------------------------------
// generate_random_multiexponentiation
//--------------------------------------------------------------------------------------------------
void generate_random_multiexponentiation(
    uint64_t& num_inputs, basct::span<mtxb::exponent_sequence>& exponents, basm::alloc_t alloc,
    std::mt19937& rng, const random_multiexponentiation_descriptor& descriptor) noexcept {
  //clang-format off
  SXT_RELEASE_ASSERT(descriptor.min_num_sequences <= descriptor.max_num_sequences &&
                     descriptor.min_sequence_length <= descriptor.max_sequence_length &&
                     descriptor.min_exponent_num_bytes <= descriptor.max_exponent_num_bytes &&
                     descriptor.max_exponent_num_bytes <= 32);
  //clang-format on
  auto num_sequences = std::uniform_int_distribution<size_t>{descriptor.min_num_sequences,
                                                             descriptor.max_num_sequences}(rng);
  exponents = {
      basm::allocate_array<mtxb::exponent_sequence>(alloc, num_sequences),
      num_sequences,
  };
  num_inputs = 0;
  for (size_t sequence_index = 0; sequence_index < num_sequences; ++sequence_index) {
    std::uniform_int_distribution<uint8_t> data_gen(0, 255);
    std::uniform_int_distribution<size_t> sequence_gen(descriptor.min_sequence_length,
                                                       descriptor.max_sequence_length);
    std::uniform_int_distribution<uint8_t> exponent_gen(descriptor.min_exponent_num_bytes,
                                                        descriptor.max_exponent_num_bytes);
    size_t sequence_length = sequence_gen(rng);
    uint8_t exponent_length = exponent_gen(rng);
    size_t sequence_total_num_bytes = sequence_length * exponent_length;
    num_inputs = std::max(num_inputs, sequence_length);
    auto exponent_array = alloc.allocate(sequence_total_num_bytes);
    std::generate_n(exponent_array, sequence_total_num_bytes,
                    [&]() noexcept { return static_cast<std::byte>(data_gen(rng)); });
    exponents[sequence_index] = {
        .element_nbytes = exponent_length,
        .n = sequence_length,
        .data = reinterpret_cast<const uint8_t*>(exponent_array),
    };
  }
}

void generate_random_multiexponentiation(
    basct::span<uint64_t>& inputs, basct::span<mtxb::exponent_sequence>& exponents,
    basm::alloc_t alloc, std::mt19937& rng,
    const random_multiexponentiation_descriptor& descriptor) noexcept {
  size_t num_inputs;
  generate_random_multiexponentiation(num_inputs, exponents, alloc, rng, descriptor);
  inputs = {
      basm::allocate_array<uint64_t>(alloc, num_inputs),
      num_inputs,
  };
  generate_uint64s(inputs, rng);
}

void generate_random_multiexponentiation(
    basct::span<c21t::element_p3>& inputs, basct::span<mtxb::exponent_sequence>& exponents,
    basm::alloc_t alloc, std::mt19937& rng,
    const random_multiexponentiation_descriptor& descriptor) noexcept {
  size_t num_inputs;
  generate_random_multiexponentiation(num_inputs, exponents, alloc, rng, descriptor);
  inputs = {
      basm::allocate_array<c21t::element_p3>(alloc, num_inputs),
      num_inputs,
  };
  rstrn::generate_random_elements(inputs, rng);
}
} // namespace sxt::mtxrn
