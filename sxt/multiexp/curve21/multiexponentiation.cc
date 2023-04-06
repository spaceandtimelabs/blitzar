#include "sxt/multiexp/curve21/multiexponentiation.h"

#include <algorithm>
#include <iterator>
#include <optional>

#include "sxt/base/container/blob_array.h"
#include "sxt/base/device/event.h"
#include "sxt/base/device/event_utility.h"
#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/curve21/constant/zero.h"
#include "sxt/curve21/type/element_p3.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/async_device_resource.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/multiexp/base/exponent_sequence.h"
#include "sxt/multiexp/curve21/multiexponentiation_cpu_driver.h"
#include "sxt/multiexp/curve21/multiproduct.h"
#include "sxt/multiexp/curve21/multiproducts_combination.h"
#include "sxt/multiexp/curve21/pippenger_multiproduct_solver.h"
#include "sxt/multiexp/pippenger/multiexponentiation.h"
#include "sxt/multiexp/pippenger/multiproduct_decomposition_gpu.h"

namespace sxt::mtxc21 {
//--------------------------------------------------------------------------------------------------
// async_compute_multiexponentiation_impl
//--------------------------------------------------------------------------------------------------
static xena::future<> async_compute_multiexponentiation_impl(
    c21t::element_p3& res, std::optional<basdv::event>& generators_event,
    basct::cspan<c21t::element_p3> generators, const mtxb::exponent_sequence& exponents) noexcept {
  auto num_bytes = exponents.element_nbytes;
  basdv::stream stream;
  memr::async_device_resource resource{stream};
  res = c21cn::zero_p3_v;

  // decompose exponents
  memmg::managed_array<unsigned> indexes{&resource};
  memmg::managed_array<unsigned> product_sizes(num_bytes * 8u);
  co_await mtxpi::compute_multiproduct_decomposition(indexes, product_sizes, stream, exponents);
  if (indexes.empty()) {
    res = c21cn::zero_p3_v;
    co_return;
  }

  // or_all
  basct::blob_array or_all{1, num_bytes};
  size_t bit_index = 0;
  for (size_t byte_index = 0; byte_index < num_bytes; ++byte_index) {
    uint8_t val = 0;
    for (int i = 0; i < 8; ++i) {
      val |= static_cast<uint8_t>(product_sizes[bit_index++] > 0) << i;
    }
    or_all[0][byte_index] = val;
  }

  // compute multiproduct
  auto last = std::remove(product_sizes.begin(), product_sizes.end(), 0u);
  product_sizes.shrink(static_cast<size_t>(std::distance(product_sizes.begin(), last)));
  memmg::managed_array<c21t::element_p3> products(product_sizes.size());
  if (generators_event) {
    basdv::async_wait_on_event(stream, *generators_event);
  }
  co_await async_compute_multiproduct(products, stream, generators, indexes, product_sizes);
  indexes.reset();

  // combine results
  combine_multiproducts({&res, 1}, or_all, products);
}

//--------------------------------------------------------------------------------------------------
// compute_multiexponentiation
//--------------------------------------------------------------------------------------------------
memmg::managed_array<c21t::element_p3>
compute_multiexponentiation(basct::cspan<c21t::element_p3> generators,
                            basct::cspan<mtxb::exponent_sequence> exponents) noexcept {
  pippenger_multiproduct_solver solver;
  multiexponentiation_cpu_driver driver{&solver};
  // Note: the cpu driver is non-blocking so that the future upon return the future is
  // available
  return mtxpi::compute_multiexponentiation(driver,
                                            {static_cast<const void*>(generators.data()),
                                             generators.size(), sizeof(c21t::element_p3)},
                                            exponents)
      .value()
      .as_array<c21t::element_p3>();
}

//--------------------------------------------------------------------------------------------------
// async_compute_multiexponentiation
//--------------------------------------------------------------------------------------------------
xena::future<memmg::managed_array<c21t::element_p3>>
async_compute_multiexponentiation(basct::cspan<c21t::element_p3> generators,
                                  basct::cspan<mtxb::exponent_sequence> exponents) noexcept {

  // set up generators
  std::optional<basdv::stream> generators_stream;
  std::optional<basdv::event> generators_event;
  memmg::managed_array<c21t::element_p3> generators_data{memr::get_device_resource()};
  if (!basdv::is_device_pointer(generators.data())) {
    generators_stream.emplace();
    generators_event.emplace();
    generators_data =
        memmg::managed_array<c21t::element_p3>{generators.size(), memr::get_device_resource()};
    basdv::async_copy_host_to_device(generators_data, generators, *generators_stream);
    basdv::record_event(*generators_event, *generators_stream);
    generators = {
        generators_data.data(),
        generators_data.size(),
    };
  }

  // compute individual multiexponentiations
  memmg::managed_array<c21t::element_p3> res(exponents.size());
  std::vector<xena::future<>> computations;
  computations.reserve(exponents.size());
  for (size_t i = 0; i < res.size(); ++i) {
    auto fut =
        async_compute_multiexponentiation_impl(res[i], generators_event, generators, exponents[i]);
    computations.emplace_back(std::move(fut));
  }
  for (auto& fut : computations) {
    co_await std::move(fut);
  }
  co_return std::move(res);
}

xena::future<c21t::element_p3>
async_compute_multiexponentiation(basct::cspan<c21t::element_p3> generators,
                                  const mtxb::exponent_sequence& exponents) noexcept {
  auto res = co_await async_compute_multiexponentiation(generators, {&exponents, 1});
  co_return res[0];
}
} // namespace sxt::mtxc21
