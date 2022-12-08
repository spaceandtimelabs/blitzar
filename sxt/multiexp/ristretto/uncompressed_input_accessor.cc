#include "sxt/multiexp/ristretto/uncompressed_input_accessor.h"

#include "sxt/curve21/type/element_p3.h"

namespace sxt::mtxrs {
//--------------------------------------------------------------------------------------------------
// get_element
//--------------------------------------------------------------------------------------------------
void uncompressed_input_accessor::get_element(c21t::element_p3& p, const void* data,
                                              size_t index) const noexcept {
  auto inputs = static_cast<const c21t::element_p3*>(data);
  p = inputs[index];
}
} // namespace sxt::mtxrs
