#include "sxt/base/iterator/split.h"

#include <algorithm>

#include "sxt/base/error/assert.h"
#include "sxt/base/num/divide_up.h"

namespace sxt::basit {
//--------------------------------------------------------------------------------------------------
// split
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
std::pair<index_range_iterator, index_range_iterator> split(const index_range& rng,
                                                            const split_options& options) noexcept {
  return {};
#if 0
  SXT_DEBUG_ASSERT(n > 0);
  auto delta = rng.b() - rng.a();
  auto step = std::max(basn::divide_up(delta, n), size_t{1});
  step = std::max(step, rng.min_chunk_size());
  step = std::min(step, rng.max_chunk_size());
  step = basn::divide_up(step, rng.chunk_multiple()) * rng.chunk_multiple();
  index_range_iterator first{index_range{rng.a(), rng.b()}, step};
  index_range_iterator last{index_range{rng.b(), rng.b()}, step};
  return {first, last};
#endif
}
#pragma clang diagnostic pop
} // namespace sxt::basit
