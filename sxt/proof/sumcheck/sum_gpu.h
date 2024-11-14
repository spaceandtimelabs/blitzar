#pragma once

#include <utility>

#include "sxt/base/container/span.h"
#include "sxt/base/device/property.h"
#include "sxt/execution/async/future_fwd.h"

namespace sxt::s25t {
class element;
}

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// sum_options
//--------------------------------------------------------------------------------------------------
struct sum_options {
  unsigned min_chunk_size = 1'000'000u;
  unsigned max_chunk_size = 4'000'0000;
  unsigned split_factor = unsigned(basdv::get_num_devices());
};

//--------------------------------------------------------------------------------------------------
// sum_gpu
//--------------------------------------------------------------------------------------------------
xena::future<> sum_gpu(basct::span<s25t::element> p, basct::cspan<s25t::element> mles,
                       basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                       basct::cspan<unsigned> product_terms, unsigned n) noexcept;
} // namespace sxt::prfsk
