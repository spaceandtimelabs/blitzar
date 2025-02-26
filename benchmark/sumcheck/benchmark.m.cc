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
#include <charconv>
#include <chrono>
#include <numeric>
#include <print>
#include <string_view>

#include "sxt/base/container/span.h"
#include "sxt/base/error/panic.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/proof/sumcheck/gpu_driver.h"
#include "sxt/proof/sumcheck/proof_computation.h"
#include "sxt/proof/sumcheck/reference_transcript.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/scalar25/random/element.h"
#include "sxt/scalar25/realization/field.h"

using namespace sxt;

struct params {
  unsigned n;
  unsigned degree;
  unsigned num_products;
  unsigned num_samples;
};

static bool read_params(params& p, int argc, char* argv[]) noexcept {
  if (argc != 5) {
    std::println("Usage: benchmark <n> <degree> <num_products> <num_samples>");
    return false;
  }

  std::string_view s;

  // n
  s = {argv[1]};
  if (std::from_chars(s.begin(), s.end(), p.n).ec != std::errc{}) {
    baser::panic("invalid argument for n: {}\n", s);
  }

  // degree
  s = {argv[2]};
  if (std::from_chars(s.begin(), s.end(), p.degree).ec != std::errc{}) {
    baser::panic("invalid argument for degree: {}\n", s);
  }

  // num_products
  s = {argv[3]};
  if (std::from_chars(s.begin(), s.end(), p.num_products).ec != std::errc{}) {
    baser::panic("invalid argument for num_products: {}\n", s);
  }

  // num_samples
  s = {argv[4]};
  if (std::from_chars(s.begin(), s.end(), p.num_samples).ec != std::errc{}) {
    baser::panic("invalid argument for num_samples: {}\n", s);
  }

  return true;
}

int main(int argc, char* argv[]) {
  params p;
  if (!read_params(p, argc, argv)) {
    return -1;
  }

  basn::fast_random_number_generator rng{1, 2};

  // mles
  memmg::managed_array<s25t::element> mles(p.n * p.degree * p.num_products);
  s25rn::generate_random_elements(mles, rng);

  // product_table
  memmg::managed_array<std::pair<s25t::element, unsigned>> product_table(p.num_products);
  for (unsigned product_index = 0; product_index < p.num_products; ++product_index) {
    s25rn::generate_random_element(product_table[product_index].first, rng);
    product_table[product_index].second = p.degree;
  }

  // product_terms
  memmg::managed_array<unsigned> product_terms(p.num_products * p.degree);
  std::iota(product_terms.begin(), product_terms.end(), 0);

  // benchmark
  auto num_rounds = basn::ceil_log2(p.n);
  std::println("n = {}", p.n);
  std::println("num_rounds = {}", num_rounds);
  std::println("degree = {}", p.degree);
  std::println("num_products = {}", p.num_products);
  std::println("num_samples = {}", p.num_samples);
  memmg::managed_array<s25t::element> polynomials((p.degree + 1u) * num_rounds);
  memmg::managed_array<s25t::element> evaluation_point(num_rounds);
  prft::transcript base_transcript{"abc123"};
  prfsk::reference_transcript<s25t::element> transcript{base_transcript};
  prfsk::gpu_driver<s25t::element> drv;

  // initial run
  {
    auto fut = prfsk::prove_sum<s25t::element>(polynomials, evaluation_point, transcript, drv, mles,
                                               product_table, product_terms, p.n);
    xens::get_scheduler().run();
  }

  // sample
  double elapse = 0;
  for (unsigned i = 0; i < (p.num_samples + 1u); ++i) {
    auto t1 = std::chrono::steady_clock::now();
    auto fut = prfsk::prove_sum<s25t::element>(polynomials, evaluation_point, transcript, drv, mles,
                                               product_table, product_terms, p.n);
    xens::get_scheduler().run();
    auto t2 = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
    if (i > 0) {
      elapse += static_cast<double>(duration.count()) / 1.0e6;
    }
  }
  std::println("average duration: {:.4e} seconds", elapse / p.num_samples);
  std::println("average throughput: {:.4e} seconds", p.n / (elapse / p.num_samples));

  return 0;
}
