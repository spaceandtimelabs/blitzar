#pragma once

#include <cstddef>

#include "sxt/base/container/span.h"

namespace sxt::mtxb { struct exponent_sequence; }

namespace sxt::mtxpi {
struct exponent_aggregates;

//--------------------------------------------------------------------------------------------------
// compute_exponent_aggregates
//--------------------------------------------------------------------------------------------------
void compute_exponent_aggregates(exponent_aggregates& aggregates,
                                 basct::cspan<mtxb::exponent_sequence> exponents) noexcept;
}  // namespace sxt::mtxpi
