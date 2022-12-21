#include "sxt/proof/transcript/transcript_utility.h"

#include "sxt/scalar25/operation/reduce.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prft {
//--------------------------------------------------------------------------------------------------
// challenge_value
//--------------------------------------------------------------------------------------------------
void challenge_value(s25t::element& value, transcript& trans, std::string_view label) noexcept {
  trans.challenge_bytes({reinterpret_cast<uint8_t*>(&value), sizeof(s25t::element)}, label);
  s25o::reduce32(value);
}

//--------------------------------------------------------------------------------------------------
// challenge_values
//--------------------------------------------------------------------------------------------------
void challenge_values(basct::span<s25t::element> values, transcript& trans,
                      std::string_view label) noexcept {
  trans.challenge_bytes(
      {reinterpret_cast<uint8_t*>(values.data()), values.size() * sizeof(s25t::element)}, label);
  for (auto& val : values) {
    s25o::reduce32(val);
  }
}
} // namespace sxt::prft
