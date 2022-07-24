#include "sxt/multiexp/ristretto/compressed_input_accessor.h"

#include "sxt/ristretto/base/byte_conversion.h"
#include "sxt/ristretto/type/compressed_element.h"

namespace sxt::mtxrs {
//--------------------------------------------------------------------------------------------------
// get_element
//--------------------------------------------------------------------------------------------------
void compressed_input_accessor::get_element(c21t::element_p3& p, const void* data,
                                            size_t index) const noexcept {
  auto inputs = static_cast<const rstt::compressed_element*>(data);
  rstb::from_bytes(p, inputs[index].data());
}
} // namespace sxt::mtxrs
