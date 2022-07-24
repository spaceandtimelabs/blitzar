#pragma once

#include "sxt/base/container/span.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/multiexp/ristretto/input_accessor.h"

namespace sxt::mtxrs {
//--------------------------------------------------------------------------------------------------
// precomputed_p3_input_accessor
//--------------------------------------------------------------------------------------------------
class precomputed_p3_input_accessor final : public input_accessor {
public:
  explicit precomputed_p3_input_accessor(basct::cspan<c21t::element_p3> elements) noexcept;

  void get_element(c21t::element_p3& p, const void* data, size_t index) const noexcept override;

private:
  basct::cspan<c21t::element_p3> elements_;
};
} // namespace sxt::mtxrs
