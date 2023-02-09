#include "sxt/ristretto/operation/compression.h"

#include "sxt/base/error/assert.h"
#include "sxt/curve21/type/element_p3.h"
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

//--------------------------------------------------------------------------------------------------
// batch_compress
//--------------------------------------------------------------------------------------------------
void batch_compress(basct::span<rstt::compressed_element> ex_p,
                    basct::cspan<c21t::element_p3> ex) noexcept {
  SXT_DEBUG_ASSERT(ex_p.size() == ex.size());
  for (size_t i = 0; i < ex.size(); ++i) {
    compress(ex_p[i], ex[i]);
  }
}
} // namespace sxt::rsto
