/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2025-present Space and Time Labs, Inc.
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
#pragma once

#include "sxt/base/container/stack_array.h"
#include "sxt/base/error/assert.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/coroutine.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/proof/sumcheck2/driver.h"
#include "sxt/proof/sumcheck2/polynomial_utility.h"

namespace sxt::prfsk2 {
//--------------------------------------------------------------------------------------------------
// cpu_driver
//--------------------------------------------------------------------------------------------------
template <basfld::element T> class cpu_driver final : public driver<T> {
  struct cpu_workspace final : public workspace {
    memmg::managed_array<T> mles;
    basct::cspan<std::pair<T, unsigned>> product_table;
    basct::cspan<unsigned> product_terms;
    unsigned n;
    unsigned num_variables;
  };

public:
  // driver
  xena::future<std::unique_ptr<workspace>>
  make_workspace(basct::cspan<T> mles, basct::cspan<std::pair<T, unsigned>> product_table,
                 basct::cspan<unsigned> product_terms, unsigned n) const noexcept override {
    auto res = std::make_unique<cpu_workspace>();
    res->mles = memmg::managed_array<T>{mles.begin(), mles.end()};
    res->product_table = product_table;
    res->product_terms = product_terms;
    res->n = n;
    res->num_variables = std::max(basn::ceil_log2(n), 1);
    return xena::make_ready_future<std::unique_ptr<workspace>>(std::move(res));
  }

  xena::future<> sum(basct::span<T> polynomial, workspace& ws) const noexcept override {
    auto& work = static_cast<cpu_workspace&>(ws);
    auto n = work.n;
    auto mid = 1u << (work.num_variables - 1u);
    SXT_RELEASE_ASSERT(work.n >= mid);

    auto mles = work.mles.data();
    auto product_table = work.product_table;
    auto product_terms = work.product_terms;

    for (auto& val : polynomial) {
      val = {};
    }

    // expand paired terms
    auto n1 = work.n - mid;
    for (unsigned i = 0; i < n1; ++i) {
      unsigned term_first = 0;
      for (auto [mult, num_terms] : product_table) {
        SXT_RELEASE_ASSERT(num_terms < polynomial.size());
        auto terms = product_terms.subspan(term_first, num_terms);
        SXT_STACK_ARRAY(p, num_terms + 1u, T);
        expand_products(p, mles + i, n, mid, terms);
        for (unsigned term_index = 0; term_index < p.size(); ++term_index) {
          muladd(polynomial[term_index], mult, p[term_index], polynomial[term_index]);
        }
        term_first += num_terms;
      }
    }

    // expand terms where the corresponding pair is zero (i.e. n is not a power of 2)
    for (unsigned i = n1; i < mid; ++i) {
      unsigned term_first = 0;
      for (auto [mult, num_terms] : product_table) {
        auto terms = product_terms.subspan(term_first, num_terms);
        SXT_STACK_ARRAY(p, num_terms + 1u, T);
        partial_expand_products(p, mles + i, n, terms);
        for (unsigned term_index = 0; term_index < p.size(); ++term_index) {
          muladd(polynomial[term_index], mult, p[term_index], polynomial[term_index]);
        }
        term_first += num_terms;
      }
    }

    return xena::make_ready_future();
  }

  xena::future<> fold(workspace& ws, const T& r) const noexcept override {
    auto& work = static_cast<cpu_workspace&>(ws);
    auto n = work.n;
    auto mid = 1u << (work.num_variables - 1u);
    auto num_mles = work.mles.size() / n;
    SXT_RELEASE_ASSERT(
        // clang-format off
      work.n >= mid && work.mles.size() % n == 0
        // clang-format on
    );

    auto mles = work.mles.data();
    memmg::managed_array<T> mles_p(num_mles * mid);

    T one_m_r = T::one();
    sub(one_m_r, one_m_r, r);
    auto n1 = work.n - mid;
    for (auto mle_index = 0; mle_index < num_mles; ++mle_index) {
      auto data = mles + n * mle_index;
      auto data_p = mles_p.data() + mid * mle_index;

      // fold paired terms
      for (unsigned i = 0; i < n1; ++i) {
        auto val = data[i];
        mul(val, val, one_m_r);
        muladd(val, r, data[mid + i], val);
        data_p[i] = val;
      }

      // fold terms paired with zero
      for (unsigned i = n1; i < mid; ++i) {
        auto val = data[i];
        mul(val, val, one_m_r);
        data_p[i] = val;
      }
    }

    work.n = mid;
    --work.num_variables;
    work.mles = std::move(mles_p);
    return xena::make_ready_future();
  }
};
} // namespace sxt::prfsk2
