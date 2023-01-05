#include "sxt/ristretto/operation/compression.h"

#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::rsto {
//--------------------------------------------------------------------------------------------------
// compress
//--------------------------------------------------------------------------------------------------
void compress(rstt::compressed_element& e_p, const c21t::element_p3& e) noexcept {
  rstb::to_bytes(e_p.data(), e);
}

//--------------------------------------------------------------------------------------------------
// decompress
//--------------------------------------------------------------------------------------------------
void decompress(c21t::element_p3& e_p, const rstt::compressed_element& e) noexcept {
  rstb::from_bytes(e_p, e.data());
}
} // namespace sxt::rsto
