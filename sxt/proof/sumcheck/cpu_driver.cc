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
#include "sxt/proof/sumcheck/cpu_driver.h"

#include <algorithm>
#include <iostream>

#include "sxt/base/container/stack_array.h"
#include "sxt/base/error/panic.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/execution/async/future.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/proof/sumcheck/polynomial_utility.h"
#include "sxt/scalar25/operation/mul.h"
#include "sxt/scalar25/operation/muladd.h"
#include "sxt/scalar25/operation/sub.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

namespace sxt::prfsk {
//--------------------------------------------------------------------------------------------------
// cpu_workspace
//--------------------------------------------------------------------------------------------------
namespace {
struct cpu_workspace final : public workspace {
  memmg::managed_array<s25t::element> mles;
  basct::cspan<std::pair<s25t::element, unsigned>> product_table;
  basct::cspan<unsigned> product_terms;
  unsigned n;
  unsigned num_variables;
};
} // namespace

//--------------------------------------------------------------------------------------------------
// make_workspace
//--------------------------------------------------------------------------------------------------
xena::future<std::unique_ptr<workspace>>
cpu_driver::make_workspace(basct::cspan<s25t::element> mles,
                           basct::cspan<std::pair<s25t::element, unsigned>> product_table,
                           basct::cspan<unsigned> product_terms, unsigned n) const noexcept {
  auto res = std::make_unique<cpu_workspace>();
  res->mles = memmg::managed_array<s25t::element>{mles.begin(), mles.end()};
  res->product_table = product_table;
  res->product_terms = product_terms;
  res->n = n;
  res->num_variables = std::max(basn::ceil_log2(n), 1);
  return xena::make_ready_future<std::unique_ptr<workspace>>(std::move(res));
}

//--------------------------------------------------------------------------------------------------
// sum
//--------------------------------------------------------------------------------------------------
xena::future<> cpu_driver::sum(basct::span<s25t::element> polynomial,
                               workspace& ws) const noexcept {
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
      auto terms = product_terms.subspan(term_first, num_terms);
      SXT_STACK_ARRAY(p, num_terms + 1u, s25t::element);
      expand_products(p, mles + i, n, mid, terms);
      for (unsigned term_index = 0; term_index < p.size(); ++term_index) {
        s25o::muladd(polynomial[term_index], mult, p[term_index], polynomial[term_index]);
      }
      term_first += num_terms;
    }
  }

  // expand terms where the corresponding pair is zero (i.e. n is not a power of 2)
  for (unsigned i = n1; i < mid; ++i) {
    unsigned term_first = 0;
    for (auto [mult, num_terms] : product_table) {
      auto terms = product_terms.subspan(term_first, num_terms);
      SXT_STACK_ARRAY(p, num_terms + 1u, s25t::element);
      partial_expand_products(p, mles + i, n, terms);
      for (unsigned term_index = 0; term_index < p.size(); ++term_index) {
        s25o::muladd(polynomial[term_index], mult, p[term_index], polynomial[term_index]);
      }
      term_first += num_terms;
    }
  }

  return xena::make_ready_future();
}

//--------------------------------------------------------------------------------------------------
// fold
//--------------------------------------------------------------------------------------------------
xena::future<> cpu_driver::fold(workspace& ws, const s25t::element& r) const noexcept {
  using s25t::operator""_s25;

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
  s25t::element one_m_r = 0x1_s25;
  s25o::sub(one_m_r, one_m_r, r);
  auto n1 = work.n - mid;
  for (auto mle_index = 0; mle_index < num_mles; ++mle_index) {
    auto data = mles + n * mle_index;

    // fold paired terms
    for (unsigned i = 0; i < n1; ++i) {
      auto val = data[i];
      s25o::mul(val, val, one_m_r);
      s25o::muladd(val, r, data[mid + i], val);
      data[i] = val;
    }

    // fold terms paired with zero
    for (unsigned i = n1; i < mid; ++i) {
      auto val = data[i];
      s25o::mul(val, val, one_m_r);
      data[i] = val;
    }
  }

  work.n = mid;
  --work.num_variables;
  work.mles.shrink(num_mles * mid);
  return xena::make_ready_future();
}
} // namespace sxt::prfsk
