#pragma once

#include "sxt/base/macro/cuda_callable.h"
#include "sxt/base/num/fast_random_number_generator.h"

#include "sxt/field51/base/byte_conversion.h"
#include "sxt/field51/type/element.h"

namespace sxt::f51rn {
//--------------------------------------------------------------------------------------------------
// generate_random_element
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
inline void generate_random_element(f51t::element& e,
                                    basn::fast_random_number_generator& generator) noexcept {
  uint64_t data[4];
  data[0] = generator();
  data[1] = generator();
  data[2] = generator();
  data[3] = generator();
  f51b::from_bytes(e.data(), reinterpret_cast<const uint8_t*>(data));
}
} // namespace sxt::f51rn
