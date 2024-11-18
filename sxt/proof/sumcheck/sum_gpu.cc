#include "sxt/proof/sumcheck/sum_gpu.h"

#include "sxt/algorithm/reduction/kernel_fit.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/execution/kernel/kernel_dims.h"
#include "sxt/execution/kernel/launch.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/proof/sumcheck/device_cache.h"
#include "sxt/proof/sumcheck/mle_utility.h"
#include "sxt/proof/sumcheck/reduction_gpu.h"
#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/type/element.h"
#include <locale>

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// partial_sum_kernel 
//--------------------------------------------------------------------------------------------------
template <unsigned BlockSize>
__global__ static void
partial_sum_kernel(s25t::element* __restrict__ out,
                   const std::pair<s25t::element, unsigned>* __restrict__ product_table,
                   const unsigned* __restrict__ product_terms, unsigned n) noexcept {}

//--------------------------------------------------------------------------------------------------
// partial_sum 
//--------------------------------------------------------------------------------------------------
static xena::future<> partial_sum(basct::span<s25t::element> p, basdv::stream& stream,
                                  basct::cspan<s25t::element> mles,
                                  basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                                  basct::cspan<unsigned> product_terms, unsigned split,
                                  unsigned n) noexcept {
  auto num_coefficients = p.size();
  auto dims = algr::fit_reduction_kernel(n);
  memr::async_device_resource resource{stream};

  // partials
  memmg::managed_array<s25t::element> partials{num_coefficients * dims.num_blocks, &resource};
  xenk::launch_kernel(dims.block_size, [&]<unsigned BlockSize>(
                                           std::integral_constant<unsigned, BlockSize>) noexcept {
#if 0
    partial_sum_kernel<BlockSize>
        <<<dim3(dims.num_blocks, num_coefficients, 1), BlockSize, 0, stream>>>(
            partials.data(), product_table.data(), product_terms.data(), n);
#endif
  });

  // reduce partials
  co_await reduce_sums(p, stream, partials);
}

//--------------------------------------------------------------------------------------------------
// sum_gpu 
//--------------------------------------------------------------------------------------------------
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-parameter"
xena::future<> sum_gpu(basct::span<s25t::element> p, device_cache& cache,
                       basct::cspan<s25t::element> mles, unsigned n) noexcept {
  auto mid = n / 2u;
  auto num_mles = mles.size() / n;
  auto num_coefficients = p.size();

  // split
  sum_options options;
  auto [chunk_first, chunk_last] = basit::split(basit::index_range{0, mid}
                                                    .min_chunk_size(options.min_chunk_size)
                                                    .max_chunk_size(options.max_chunk_size),
                                                options.split_factor);

  // sum
  size_t counter = 0;
  co_await xendv::concurrent_for_each(
      chunk_first, chunk_last, [&](const basit::index_range& rng) noexcept -> xena::future<> {
        basdv::stream stream;
        memr::async_device_resource resource{stream};

        // copy partial mles to device
        memmg::managed_array<s25t::element> partial_mles{&resource};
        copy_partial_mles(partial_mles, stream, mles, n, rng.a(), rng.b());
        auto split = rng.b() - rng.a();
        auto np = partial_mles.size() / num_mles;

        // lookup problem descriptor
        basct::cspan<std::pair<s25t::element, unsigned>> product_table;
        basct::cspan<unsigned> product_terms;
        cache.lookup(product_table, product_terms, stream);

        // compute
        memmg::managed_array<s25t::element> partial_p(num_coefficients);
        co_await partial_sum(partial_p, stream, partial_mles, product_table, product_terms, split,
                             np);

        // fill in the result
        if (counter == 0) {
          for (unsigned i = 0; i < num_coefficients; ++i) {
            p[i] = partial_p[i];
          }
        } else {
          for (unsigned i = 0; i < num_coefficients; ++i) {
            s25o::add(p[i], p[i], partial_p[i]);
          }
        }
        ++counter;
      });
}
#pragma clang diagnostic pop
} // namespace sxt::prfsk
