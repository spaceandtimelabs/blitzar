#include "sxt/ristretto/operation/scalar_multiply.h"

#include "sxt/curve21/operation/scalar_multiply.h"
#include "sxt/curve21/type/element_p3.h"

#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::rsto {
//--------------------------------------------------------------------------------------------------
// scalar_multiply
//--------------------------------------------------------------------------------------------------
/*
 h = a * p
 where a = a[0]+256*a[1]+...+256^31 a[31]
 */
CUDA_CALLABLE
void scalar_multiply(rstt::compressed_element& r, basct::cspan<uint8_t> a,
                     const rstt::compressed_element& p) noexcept {

  c21t::element_p3 temp_p;

  rstb::from_bytes(temp_p, p.data());

  c21o::scalar_multiply(temp_p, a, temp_p);

  rstb::to_bytes(r.data(), temp_p);
}
} // namespace sxt::rsto
