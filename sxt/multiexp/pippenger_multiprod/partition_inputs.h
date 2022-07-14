#pragma once

#include <cstddef>
#include <cstdint>

#include "sxt/base/container/span.h"

namespace sxt::basct { class span_void; }
namespace sxt::mtxi { class index_table; }

namespace sxt::mtxpmp {
class driver;
struct reduction_stats;

//--------------------------------------------------------------------------------------------------
// partition_inputs
//--------------------------------------------------------------------------------------------------
void partition_inputs(basct::span_void inout, reduction_stats& stats,
                      basct::span<basct::span<uint64_t>> products,
                      size_t& num_inactive_outputs, size_t& num_inactive_inputs,
                      const driver& drv, size_t partition_size) noexcept;
} // namespace sxt::mtxpmp
