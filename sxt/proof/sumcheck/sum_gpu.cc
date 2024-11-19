#include "sxt/proof/sumcheck/sum_gpu.h"

#include <cstddef>

#include "sxt/algorithm/reduction/kernel_fit.h"
#include "sxt/algorithm/reduction/thread_reduction.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/iterator/index_range_iterator.h"
#include "sxt/base/iterator/index_range_utility.h"
#include "sxt/base/num/constexpr_switch.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/device/for_each.h"
#include "sxt/execution/kernel/kernel_dims.h"
#include "sxt/execution/kernel/launch.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/proof/sumcheck/constant.h"
#include "sxt/proof/sumcheck/device_cache.h"
#include "sxt/proof/sumcheck/mle_utility.h"
#include "sxt/proof/sumcheck/polynomial_mapper2.h"
#include "sxt/proof/sumcheck/reduction_gpu.h"
#include "sxt/scalar25/operation/add.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/type/element.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// polynomial_reducer
//--------------------------------------------------------------------------------------------------
namespace {
template <unsigned Degree> struct polynomial_reducer {
  using value_type = std::array<s25t::element, Degree + 1u>;

  CUDA_CALLABLE static void accumulate_inplace(value_type& res, const value_type& e) noexcept {
    for (unsigned i = 0; i < res.size(); ++i) {
      s25o::add(res[i], res[i], e[i]);
    }
  }
};
} // namespace

//--------------------------------------------------------------------------------------------------
// partial_sum_kernel_impl 
//--------------------------------------------------------------------------------------------------
template <unsigned BlockSize, unsigned NumTerms>
__device__ static void partial_sum_kernel_impl(s25t::element* __restrict__ out,
                                               s25t::element* __restrict__ shared_data,
                                               const s25t::element* __restrict__ mles,
                                               const unsigned* __restrict__ product_terms,
                                               unsigned split, unsigned n) noexcept {
  using Mapper = polynomial_mapper2<NumTerms>;
  using Reducer = polynomial_reducer<NumTerms>;
  using T = Mapper::value_type;
  Mapper mapper{
      .mles = mles,
      .product_terms = product_terms,
      .split = split,
      .n = n,
  };
  auto index = blockIdx.x * (BlockSize * 2) + threadIdx.x;
  auto step = BlockSize * 2 * gridDim.x;
  algr::thread_reduce<Reducer, BlockSize>(reinterpret_cast<T*>(out),
                                          reinterpret_cast<T*>(shared_data), mapper, split, step,
                                          threadIdx.x, index);
  /* template <algb::reducer Reducer, unsigned int BlockSize, algb::mapper Mapper> */
  /*   requires std::same_as<typename Reducer::value_type, typename Mapper::value_type> */
  /* __device__ void thread_reduce(typename Reducer::value_type* out, */
  /*                               typename Reducer::value_type* shared_data, Mapper mapper, */
  /*                               unsigned int n, unsigned int step, unsigned int thread_index, */
  /*                               unsigned int index) { */
}

//--------------------------------------------------------------------------------------------------
// partial_sum_kernel 
//--------------------------------------------------------------------------------------------------
template <unsigned BlockSize>
__global__ static void
partial_sum_kernel(s25t::element* __restrict__ out, const s25t::element* __restrict__ mles,
                   const std::pair<s25t::element, unsigned>* __restrict__ product_table,
                   const unsigned* __restrict__ product_terms, unsigned num_coefficients,
                   unsigned split, unsigned n) noexcept {
  auto term_index = blockIdx.y;
  auto num_terms = product_table[term_index].second;
  auto thread_index = threadIdx.x;

  // shared data for reduction
  __shared__ s25t::element shared_data[2 * BlockSize * (max_degree_v + 1u)];

  // adjust pointers
  out += num_coefficients * term_index;
  for (unsigned i=0; i<term_index; ++i) {
    product_terms += product_table[i].second;
  }

  // sum
  basn::constexpr_switch<1, max_degree_v>(
      num_terms, [&]<unsigned NumTerms>(std::integral_constant<unsigned, NumTerms>) noexcept {
        partial_sum_kernel_impl<BlockSize, NumTerms>(out, shared_data, mles, product_terms, split,
                                                     n);
      });

  // write out result
  if (blockIdx.x != 0) {
    return;
  }
  auto mult = product_table[term_index].first;
  for (unsigned i = thread_index; i < num_coefficients; i += BlockSize) {
    if (i < num_terms + 1u) {
      s25o::mul(out[i], mult, out[i]);
    } else {
      out[i] = s25t::element{};
    }
  }
}

//--------------------------------------------------------------------------------------------------
// partial_sum 
//--------------------------------------------------------------------------------------------------
static xena::future<> partial_sum(basct::span<s25t::element> p, basdv::stream& stream,
                                  basct::cspan<s25t::element> mles,
                                  basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                                  basct::cspan<unsigned> product_terms, unsigned split,
                                  unsigned n) noexcept {
  auto num_coefficients = p.size();
  auto dims = algr::fit_reduction_kernel(split);
  memr::async_device_resource resource{stream};

  // partials
  memmg::managed_array<s25t::element> partials{num_coefficients * dims.num_blocks, &resource};
  xenk::launch_kernel(dims.block_size, [&]<unsigned BlockSize>(
                                           std::integral_constant<unsigned, BlockSize>) noexcept {
    partial_sum_kernel<BlockSize>
        <<<dim3(dims.num_blocks, num_coefficients, 1), BlockSize, 0, stream>>>(
            partials.data(), mles.data(), product_table.data(), product_terms.data(),
            num_coefficients, split, n);
  });

  // reduce partials
  co_await reduce_sums(p, stream, partials);
}

//--------------------------------------------------------------------------------------------------
// sum_gpu 
//--------------------------------------------------------------------------------------------------
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
} // namespace sxt::prfsk
