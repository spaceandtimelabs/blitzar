#pragma once

#include <cstddef>
#include <limits>

#include "sxt/base/device/property.h"

namespace sxt::basit {
//--------------------------------------------------------------------------------------------------
// chunk_options
//--------------------------------------------------------------------------------------------------
struct chunk_options {
  size_t min_size = 1;
  size_t max_size = std::numeric_limits<size_t>::max();
  size_t split_factor = basdv::get_num_devices();
};
} // namespace sxt::basit
