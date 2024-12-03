#include <print>
#include <random>

#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/memory/management/managed_array.h"
using namespace sxt;

int main() {
  const unsigned n = 1'000'00;
  const unsigned m = 32;
  const unsigned split_factor = 16;

  // fill data
  memmg::managed_array<double> data(n * m);
  std::mt19937 rng{0};
  std::uniform_real_distribution<double> dist{-1, 1};
  for (auto& x : data) {
    x = dist(rng);
  }
  (void)data;

  memmg::managed_array<double> sum(n);
  auto [chunk_first, chunk_last] = basit::split(basit::index_range{0, n}, split_factor);

  // 1 call repeat copies from ordinary memory to device
  auto fut = xendv::concurrent_for_each(
      chunk_first, chunk_last,
      [&](const basit::index_range& rng) noexcept -> xena::future<> { co_return; });
  // 2 copy to contiguous paged memory, copy from paged memory to device
  // 3 like 2, but use chunks
  std::println("arf");
  return 0;
}
