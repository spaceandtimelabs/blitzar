#pragma once

#include "sxt/base/macro/cuda_callable.h"

namespace sxt::c21t {
struct element_p3;
}
namespace sxt::f51t {
class element;
}

namespace sxt::rstb {
//--------------------------------------------------------------------------------------------------
// form_ristretto_point
//--------------------------------------------------------------------------------------------------
CUDA_CALLABLE
void form_ristretto_point(c21t::element_p3& p, const f51t::element& r0,
                          const f51t::element& r1) noexcept;
} // namespace sxt::rstb
