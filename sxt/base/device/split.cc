#include "sxt/base/device/split.h"

#include "sxt/base/device/property.h"

namespace sxt::basdv {
//--------------------------------------------------------------------------------------------------
// plan_split
//--------------------------------------------------------------------------------------------------
basit::split_options plan_split(size_t bytes) noexcept {
  auto device_memory = get_total_device_memory();

  auto high_memory_target = device_memory / 4u;
  auto low_memory_target = device_memory / 64u;

  auto high_target = high_memory_target / bytes;
  auto low_target = low_memory_target / bytes;

  high_target = std::max<size_t>(1u, high_target);
  low_target = std::max<size_t>(1u, low_target);

  return basit::split_options{
    .min_chunk_size = low_target,
    .max_chunk_size = high_target,
    .split_factor = basdv::get_num_devices(),
  };
}
} // namespace sxt::basdv
