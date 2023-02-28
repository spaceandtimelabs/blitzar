#pragma once

#include "sxt/base/container/span.h"
#include "sxt/memory/management/managed_array_fwd.h"

namespace sxt::basct {
class blob_array;
}
namespace sxt::mtxb {
struct exponent_sequence;
}

namespace sxt::mtxpi {
//--------------------------------------------------------------------------------------------------
// compute_multiproduct_decomposition
//--------------------------------------------------------------------------------------------------
void compute_multiproduct_decomposition(memmg::managed_array<unsigned>& indexes,
                                        memmg::managed_array<unsigned>& product_sizes,
                                        basct::blob_array& output_digit_or_all,
                                        basct::cspan<mtxb::exponent_sequence> exponents) noexcept;
} // namespace sxt::mtxpi
