#pragma once

#include <algorithm>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/execution/device/synchronization.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/proof/sumcheck2/driver.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// gpu_driver
//--------------------------------------------------------------------------------------------------
template <basfld::element T>
class gpu_driver final : public driver<T> {
public:
  struct gpu_workspace final : public workspace {
    memmg::managed_array<T> mles;
    memmg::managed_array<std::pair<T, unsigned>> product_table;
    memmg::managed_array<unsigned> product_terms;
    unsigned n;
    unsigned num_variables;

    gpu_workspace() noexcept
        : mles{memr::get_device_resource()}, product_table{memr::get_device_resource()},
          product_terms{memr::get_device_resource()} {}
  };

  // driver
  xena::future<std::unique_ptr<workspace>>
  make_workspace(basct::cspan<T> mles,
                 basct::cspan<std::pair<T, unsigned>> product_table,
                 basct::cspan<unsigned> product_terms, unsigned n) const noexcept override {
    auto ws = std::make_unique<gpu_workspace>();

    // dimensions
    ws->n = n;
    ws->num_variables = std::max(basn::ceil_log2(n), 1);

    // mles
    ws->mles = memmg::managed_array<T>{
        mles.size(),
        memr::get_device_resource(),
    };
    basdv::stream mle_stream;
    basdv::async_copy_host_to_device(ws->mles, mles, mle_stream);

    // product_table
    ws->product_table = memmg::managed_array<std::pair<T, unsigned>>{
        product_table.size(),
        memr::get_device_resource(),
    };
    basdv::stream product_table_stream;
    basdv::async_copy_host_to_device(ws->product_table, product_table, product_table_stream);

    // product_terms
    ws->product_terms = memmg::managed_array<unsigned>{
        product_terms.size(),
        memr::get_device_resource(),
    };
    basdv::stream product_terms_stream;
    basdv::async_copy_host_to_device(ws->product_terms, product_terms, product_terms_stream);

    // await
    co_await xendv::await_stream(mle_stream);
    co_await xendv::await_stream(product_table_stream);
    co_await xendv::await_stream(product_terms_stream);
    co_return ws;
  }

  xena::future<> sum(basct::span<T> polynomial, workspace& ws) const noexcept override {
    return {};
  }

  xena::future<> fold(workspace& ws, const T& r) const noexcept override {
    return {};
  }
};
} // namespace sxt::prfsk2
