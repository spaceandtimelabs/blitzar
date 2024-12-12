/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2024-present Space and Time Labs, Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <chrono>
#include <cstring>
#include <print>
#include <random>

#include "sxt/algorithm/iteration/for_each.h"
#include "sxt/base/container/span.h"
#include "sxt/base/container/span_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/pinned_buffer_pool.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/iterator/split.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/copy.h"
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
  for (unsigned i = 0; i < m; ++i) {
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

// sum3
static xena::future<> sum3(basct::span<double> res, basct::cspan<double> data, unsigned n,
                           unsigned a) noexcept {
  auto chunk_size = res.size();
  auto m = data.size() / n;

  basdv::stream stream;
  memr::async_device_resource resource{stream};

  // copy
  memmg::managed_array<double> data_dev{res.size() * m, &resource};
  co_await xendv::strided_copy_host_to_device<double>(data_dev, stream, data, n, chunk_size, a);

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

using benchmark_fn = xena::future<> (*)(basct::span<double>, basct::cspan<double>, unsigned,
                                        unsigned);

// run_benchmark
static double run_benchmark(benchmark_fn f, unsigned n, unsigned m,
                            unsigned split_factor) noexcept {
  // fill data
  memmg::managed_array<double> data(n * m);
  std::mt19937 rng{0};
  std::uniform_real_distribution<double> dist{-1, 1};
  for (auto& x : data) {
    x = dist(rng);
  }

  // chunk
  basit::split_options split_options{
    .split_factor = split_factor,
  };
  auto [chunk_first, chunk_last] = basit::split(basit::index_range{0, n}, split_options);

  // invoker
  memmg::managed_array<double> sum(n);
  auto invoker = [&] {
    auto fut = xendv::concurrent_for_each(
        chunk_first, chunk_last, [&](const basit::index_range& rng) noexcept -> xena::future<> {
          co_await f(basct::subspan(sum, rng.a(), rng.size()), data, n, rng.a());
        });
    xens::get_scheduler().run();
  };

  // initial run
  invoker();

  // average
  auto avg = 0.0;
  unsigned num_iterations = 10;
  for (unsigned i = 0; i < num_iterations; ++i) {
    auto t1 = std::chrono::steady_clock::now();
    invoker();
    auto t2 = std::chrono::steady_clock::now();
    auto elapse = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count() / 1.0e3;
    avg += elapse;
  }
  return avg / num_iterations;
}

int main() {
  const unsigned n = 1'000'00;
  const unsigned m = 32;
  const unsigned split_factor = 16;

  auto avg_elapse = run_benchmark(sum1, n, m, split_factor);
  std::println("sum1: average elapse: {}", avg_elapse);

  avg_elapse = run_benchmark(sum2, n, m, split_factor);
  std::println("sum2: average elapse: {}", avg_elapse);

  avg_elapse = run_benchmark(sum3, n, m, split_factor);
  std::println("sum3: average elapse: {}", avg_elapse);

  return 0;
}
