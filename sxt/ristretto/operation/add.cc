#include "sxt/ristretto/operation/add.h"

#include "sxt/curve21/operation/add.h"
#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::rsto {
//--------------------------------------------------------------------------------------------------
// add
//--------------------------------------------------------------------------------------------------
/*
 r = p + q
 */
CUDA_CALLABLE
void add(rstt::compressed_element& r, const rstt::compressed_element& p,
         const rstt::compressed_element& q) noexcept {

  c21t::element_p3 temp_p, temp_q;

  rstb::from_bytes(temp_p, p.data());
  rstb::from_bytes(temp_q, q.data());

  c21o::add(temp_p, temp_p, temp_q);

  rstb::to_bytes(r.data(), temp_p);
}
} // namespace sxt::rsto
