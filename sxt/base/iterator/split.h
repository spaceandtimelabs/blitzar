#pragma once

#include <limits>
#include <utility>

#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"

namespace sxt::basit {
//--------------------------------------------------------------------------------------------------
// split_options
//--------------------------------------------------------------------------------------------------
struct split_options {
  size_t min_chunk_size = 1;
  size_t max_chunk_size = std::numeric_limits<size_t>::max();
  size_t split_factor = 1;
};

//--------------------------------------------------------------------------------------------------
// split
//--------------------------------------------------------------------------------------------------
std::pair<index_range_iterator, index_range_iterator> split(const index_range& rng,
                                                            const split_options& options) noexcept;
} // namespace sxt::basit
