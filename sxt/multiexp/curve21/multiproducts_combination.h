#pragma once

#include "sxt/base/container/span.h"

namespace sxt::basct {
class blob_array;
}
namespace sxt::c21t {
struct element_p3;
}

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// combine_multiproducts
//--------------------------------------------------------------------------------------------------
void combine_multiproducts(basct::span<c21t::element_p3> outputs,
                           const basct::blob_array& output_digit_or_all,
                           basct::cspan<c21t::element_p3> products) noexcept;
} // namespace sxt::mtxc21
