#include "sxt/base/iterator/split.h"

#include <algorithm>

#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"

namespace sxt::basit {
//--------------------------------------------------------------------------------------------------
// split
//--------------------------------------------------------------------------------------------------
std::pair<index_range_iterator, index_range_iterator> split(const index_range& rng,
                                                            const split_options& options) noexcept {
  SXT_DEBUG_ASSERT(options.split_factor > 0);
  auto delta = rng.b() - rng.a();
  auto step = std::max(basn::divide_up(delta, options.split_factor), size_t{1});
  step = std::max(step, options.min_chunk_size);
  step = std::min(step, options.max_chunk_size);
  step = basn::divide_up(step, rng.chunk_multiple()) * rng.chunk_multiple();
  index_range_iterator first{index_range{rng.a(), rng.b()}, step};
  index_range_iterator last{index_range{rng.b(), rng.b()}, step};
  return {first, last};
}
} // namespace sxt::basit
