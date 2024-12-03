#include <print>
#include <random>
#include <chrono>
#include <cstring>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/iterator/index_range.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/pinned_resource.h"
using namespace sxt;

// sum1
static xena::future<> sum1(basct::span<double> res, basct::cspan<double> data, unsigned n,
                           unsigned a) noexcept {
  auto chunk_size = res.size();
  auto m = data.size() / n;

  basdv::stream stream;
  memr::async_device_resource resource{stream};
  
  // copy
  memmg::managed_array<double> data_dev{res.size() * m, &resource};
  for (unsigned i=0; i<m; ++i) {
    basdv::async_copy_host_to_device(basct::subspan(data_dev, i * chunk_size, chunk_size),
                                     basct::subspan(data, i * n + a, chunk_size), stream);
  }

  // sum
  memmg::managed_array<double> res_dev{chunk_size, &resource};
  auto f = [res = res_dev.data(), data = data_dev.data(), m = m] __device__ __host__(
               unsigned chunk_size, unsigned i) noexcept {
    double sum = 0;
    for (unsigned j = 0; j < m; ++j) {
      sum += data[i + chunk_size * j];
    }
    res[i] = sum;
  };
  algi::launch_for_each_kernel(stream, f, chunk_size);
  basdv::async_copy_device_to_host(res, res_dev, stream);
  co_await xendv::await_stream(stream);
}

// sum2
static xena::future<> sum2(basct::span<double> res, basct::cspan<double> data, unsigned n,
                           unsigned a) noexcept {
  auto chunk_size = res.size();
  auto m = data.size() / n;

  basdv::stream stream;
  memr::async_device_resource resource{stream};
  
  // copy
  /* memmg::managed_array<double> data_p{res.size() * m, memr::get_pinned_resource()}; */
  memmg::managed_array<double> data_p(res.size() * m);
  memmg::managed_array<double> data_dev{res.size() * m, &resource};
  for (unsigned i = 0; i < m; ++i) {
    std::memcpy(static_cast<void*>(data_p.data() + chunk_size * i),
                static_cast<const void*>(data.data() + a + n * i), chunk_size * sizeof(double));
  }
  basdv::async_copy_host_to_device(data_dev, data_p, stream);

  // sum
  memmg::managed_array<double> res_dev{chunk_size, &resource};
  auto f = [res = res_dev.data(), data = data_dev.data(), m = m] __device__ __host__(
               unsigned chunk_size, unsigned i) noexcept {
    double sum = 0;
    for (unsigned j = 0; j < m; ++j) {
      sum += data[i + chunk_size * j];
    }
    res[i] = sum;
  };
  algi::launch_for_each_kernel(stream, f, chunk_size);
  basdv::async_copy_device_to_host(res, res_dev, stream);
  co_await xendv::await_stream(stream);
}

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
  auto t1 = std::chrono::steady_clock::now();
  auto fut = xendv::concurrent_for_each(
      chunk_first, chunk_last, [&](const basit::index_range& rng) noexcept -> xena::future<> {
#if 0
        co_await sum1(basct::subspan(sum, rng.a(), rng.size()), data, n, rng.a());
#else
        co_await sum2(basct::subspan(sum, rng.a(), rng.size()), data, n, rng.a());
#endif
      });
  xens::get_scheduler().run();
  auto t2 = std::chrono::steady_clock::now();
  auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1.0e3;
  std::println("duration: {}", elapse);
  std::println("sums {} ... {}", sum[0], sum[n-1]);

  // 2 copy to contiguous paged memory, copy from paged memory to device
  // 3 like 2, but use chunks
  return 0;
}
