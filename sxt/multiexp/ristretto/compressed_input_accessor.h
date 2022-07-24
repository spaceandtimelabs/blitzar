#pragma once

#include <cstddef>

#include "sxt/multiexp/ristretto/input_accessor.h"

namespace sxt::mtxrs {
//--------------------------------------------------------------------------------------------------
// compressed_input_accessor
//--------------------------------------------------------------------------------------------------
class compressed_input_accessor final : public input_accessor {
public:
  void get_element(c21t::element_p3& p, const void* data, size_t index) const noexcept override;
};
} // namespace sxt::mtxrs
