#pragma once

#include <utility>

#include "sxt/base/container/span.h"
#include "sxt/base/device/property.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::s25t {
class element;
}

namespace sxt::prfsk {
class device_cache;

//--------------------------------------------------------------------------------------------------
// sum_options
//--------------------------------------------------------------------------------------------------
struct sum_options {
  unsigned min_chunk_size = 100'000u;
  unsigned max_chunk_size = 1'000'000u;
  unsigned split_factor = unsigned(basdv::get_num_devices());
};

//--------------------------------------------------------------------------------------------------
// sum_gpu
//--------------------------------------------------------------------------------------------------
xena::future<> sum_gpu(basct::span<s25t::element> p, device_cache& cache,
                       const sum_options& options, basct::cspan<s25t::element> mles,
                       unsigned n) noexcept;

xena::future<> sum_gpu(basct::span<s25t::element> p, device_cache& cache,
                       basct::cspan<s25t::element> mles, unsigned n) noexcept;
} // namespace sxt::prfsk
