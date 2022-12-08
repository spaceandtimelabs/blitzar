#pragma once

#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::c21t {
struct element_p3;
}

namespace sxt::mtxrs {
//--------------------------------------------------------------------------------------------------
// doubling_reduce
//--------------------------------------------------------------------------------------------------
void doubling_reduce(c21t::element_p3& res, basct::cspan<uint8_t> digit_or_all,
                     basct::cspan<c21t::element_p3> inputs) noexcept;
} // namespace sxt::mtxrs
