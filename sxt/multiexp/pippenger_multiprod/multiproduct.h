#pragma once

#include <cstddef>
#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::basct { class span_void; }
namespace sxt::mtxi { class index_table; }

namespace sxt::mtxpmp {
class driver;
struct multiproduct_params;

//--------------------------------------------------------------------------------------------------
// compute_multiproduct
//--------------------------------------------------------------------------------------------------
void compute_multiproduct(basct::span_void inout,
                          basct::span<basct::span<uint64_t>> products,
                          const driver& drv, size_t num_inputs) noexcept;

void compute_multiproduct(basct::span_void inout, mtxi::index_table& products,
                          const driver& drv, size_t num_inputs) noexcept;
}  // namespace sxt::mtxpmp
