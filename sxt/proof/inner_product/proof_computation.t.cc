/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023 Space and Time Labs, Inc.
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
#include "sxt/proof/inner_product/proof_computation.h"

#include <cstdlib>
#include <iostream>
#include <memory_resource>

#include "sxt/base/error/panic.h"
#include "sxt/base/num/ceil_log2.h"
#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/overload.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/proof/inner_product/cpu_driver.h"
#include "sxt/proof/inner_product/gpu_driver.h"
#include "sxt/proof/inner_product/proof_descriptor.h"
#include "sxt/proof/inner_product/random_product_generation.h"
#include "sxt/proof/transcript/transcript.h"
#include "sxt/ristretto/operation/compression.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/element.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::prfip;
using sxt::s25t::operator""_s25;
static void exercise_prove_verify(const driver& drv, const proof_descriptor& descriptor,
                                  basct::cspan<s25t::element> a_vector) noexcept;

TEST_CASE("we can prove and verify an inner product") {
  std::pmr::monotonic_buffer_resource alloc;
  static cpu_driver cpu_drv;
  static gpu_driver gpu_drv;
  const driver& drv = *GENERATE_COPY(&cpu_drv, &gpu_drv);
  proof_descriptor descriptor;
  basct::cspan<s25t::element> a_vector;
  basn::fast_random_number_generator rng{1, 2};

  s25t::element ap_value;
  prft::transcript transcript{"abc"};

  auto& b_vector = descriptor.b_vector;
  auto& g_vector = descriptor.g_vector;

  SECTION("we can prove and verify with a single element") {
    generate_random_product(descriptor, a_vector, rng, &alloc, 1);
    std::vector<rstt::compressed_element> l_vector, r_vector;
    auto p_fut =
        prove_inner_product(l_vector, r_vector, ap_value, transcript, drv, descriptor, a_vector);
    xens::get_scheduler().run();
    REQUIRE(p_fut.ready());

    // Note: transcript not used for single element proof
    auto product = a_vector[0] * b_vector[0];
    auto a_commit = a_vector[0] * g_vector[0];
    auto v_fut = verify_inner_product(transcript, drv, descriptor, product, a_commit, l_vector,
                                      r_vector, ap_value);
    xens::get_scheduler().run();
    REQUIRE(v_fut.value());

    auto a_commit_p = 0x123_s25 * g_vector[0];
    v_fut = verify_inner_product(transcript, drv, descriptor, product, a_commit_p, l_vector,
                                 r_vector, ap_value);
    xens::get_scheduler().run();
    REQUIRE(!v_fut.value());

    auto product_p = product + 0x123_s25;
    v_fut = verify_inner_product(transcript, drv, descriptor, product_p, a_commit, l_vector,
                                 r_vector, ap_value);
    xens::get_scheduler().run();
    REQUIRE(!v_fut.value());
  }

  SECTION("we can verify a proof with 2 elements") {
    generate_random_product(descriptor, a_vector, rng, &alloc, 2);
    std::vector<rstt::compressed_element> l_vector(1), r_vector(1);
    auto p_fut =
        prove_inner_product(l_vector, r_vector, ap_value, transcript, drv, descriptor, a_vector);
    xens::get_scheduler().run();
    REQUIRE(p_fut.ready());

    transcript = prft::transcript{"abc"};
    auto product = a_vector[0] * b_vector[0] + a_vector[1] * b_vector[1];
    auto a_commit = a_vector[0] * g_vector[0] + a_vector[1] * descriptor.g_vector[1];
    auto v_fut = verify_inner_product(transcript, drv, descriptor, product, a_commit, l_vector,
                                      r_vector, ap_value);
    xens::get_scheduler().run();
    REQUIRE(v_fut.value());

    transcript = prft::transcript{"abc"};
    auto a_commit_p = 0x123_s25 * g_vector[0];
    v_fut = verify_inner_product(transcript, drv, descriptor, product, a_commit_p, l_vector,
                                 r_vector, ap_value);
    xens::get_scheduler().run();
    REQUIRE(!v_fut.value());

    transcript = prft::transcript{"abc"};
    auto product_p = product + 0x123_s25;
    v_fut = verify_inner_product(transcript, drv, descriptor, product_p, a_commit, l_vector,
                                 r_vector, ap_value);
    xens::get_scheduler().run();
    REQUIRE(!v_fut.value());
  }

  SECTION("we can prove and verify random proofs of varying size") {
    for (size_t n = 1; n <= 9; ++n) {
      generate_random_product(descriptor, a_vector, rng, &alloc, n);
      exercise_prove_verify(drv, descriptor, a_vector);
    }
  }

  SECTION("we can prove and verify a random proof of larger size") {
    size_t n = 123;
    generate_random_product(descriptor, a_vector, rng, &alloc, n);
    exercise_prove_verify(drv, descriptor, a_vector);
  }
}

static void exercise_prove_verify(const driver& drv, const proof_descriptor& descriptor,
                                  basct::cspan<s25t::element> a_vector) noexcept {
  auto n = a_vector.size();
  auto num_rounds = basn::ceil_log2(a_vector.size());

  auto& b_vector = descriptor.b_vector;
  auto& g_vector = descriptor.g_vector;

  // create proof
  std::vector<rstt::compressed_element> l_vector(num_rounds), r_vector(num_rounds);
  s25t::element ap_value;
  prft::transcript transcript{"abc"};
  auto p_fut =
      prove_inner_product(l_vector, r_vector, ap_value, transcript, drv, descriptor, a_vector);
  xens::get_scheduler().run();
  REQUIRE(p_fut.ready());

  // verify proof
  s25t::element product = a_vector[0] * b_vector[0];
  c21t::element_p3 a_commit = a_vector[0] * g_vector[0];
  for (size_t i = 1; i < n; ++i) {
    product = product + a_vector[i] * b_vector[i];
    a_commit = a_commit + a_vector[i] * g_vector[i];
  }
  transcript = prft::transcript{"abc"};
  auto v_fut = verify_inner_product(transcript, drv, descriptor, product, a_commit, l_vector,
                                    r_vector, ap_value);
  xens::get_scheduler().run();
  if (!v_fut.value()) {
    baser::panic("failed to verify proof");
  }

  // verify fails if ap_value is wrong
  transcript = prft::transcript{"abc"};
  auto ap_value_p = ap_value + 0x5134_s25;
  v_fut = verify_inner_product(transcript, drv, descriptor, product, a_commit, l_vector, r_vector,
                               ap_value_p);
  xens::get_scheduler().run();
  if (v_fut.value()) {
    baser::panic("verification should fail");
  }

  // verify fails if product is wrong
  transcript = prft::transcript{"abc"};
  auto product_p = product + 0x5134_s25;
  v_fut = verify_inner_product(transcript, drv, descriptor, product_p, a_commit, l_vector, r_vector,
                               ap_value);
  xens::get_scheduler().run();
  if (v_fut.value()) {
    baser::panic("verification should fail");
  }

  // verification should fail if a_commit is wrong
  transcript = prft::transcript{"abc"};
  auto a_commit_p = a_commit + 0x5134_s25 * g_vector[0];
  v_fut = verify_inner_product(transcript, drv, descriptor, product, a_commit_p, l_vector, r_vector,
                               ap_value);
  xens::get_scheduler().run();
  if (v_fut.value()) {
    baser::panic("verification should fail");
  }

  // verification fails if the transcript is wrong and n > 1
  if (n > 1) {
    transcript = prft::transcript{"xyz"};
    v_fut = verify_inner_product(transcript, drv, descriptor, product, a_commit, l_vector, r_vector,
                                 ap_value);
    xens::get_scheduler().run();
    if (v_fut.value()) {
      baser::panic("verification should fail");
    }
  }

  // verification fails if the number of L or R values is wrong
  auto l_vector_p = l_vector;
  l_vector_p.emplace_back();
  transcript = prft::transcript{"abc"};
  if (verify_inner_product(transcript, drv, descriptor, product, a_commit, l_vector_p, r_vector,
                           ap_value)
          .value()) {
    baser::panic("verification should fail");
  }
  auto r_vector_p = r_vector;
  r_vector_p.emplace_back();
  transcript = prft::transcript{"abc"};
  v_fut = verify_inner_product(transcript, drv, descriptor, product, a_commit, l_vector, r_vector_p,
                               ap_value);
  xens::get_scheduler().run();
  if (v_fut.value()) {
    baser::panic("verification should fail");
  }

  // verification fails if an L or R value is wrong (n > 1)
  if (n > 1) {
    auto some_val = 0x123_s25 * g_vector[0];
    auto l_vector_p = l_vector;
    rsto::compress(l_vector_p[0], some_val);
    transcript = prft::transcript{"abc"};
    v_fut = verify_inner_product(transcript, drv, descriptor, product, a_commit, l_vector_p,
                                 r_vector, ap_value);
    xens::get_scheduler().run();
    if (v_fut.value()) {
      baser::panic("verification should fail");
    }
    auto r_vector_p = r_vector;
    rsto::compress(r_vector_p[0], some_val);
    transcript = prft::transcript{"abc"};
    v_fut = verify_inner_product(transcript, drv, descriptor, product, a_commit, l_vector,
                                 r_vector_p, ap_value);
    xens::get_scheduler().run();
    if (v_fut.value()) {
      baser::panic("verification should fail");
    }
  }

  // verification fails if b_vector is wrong
  std::vector<s25t::element> b_vector_p{b_vector.begin(), b_vector.end()};
  b_vector_p[0] = 0x123_s25;
  auto descriptor_p = descriptor;
  descriptor_p.b_vector = b_vector_p;
  transcript = prft::transcript{"abc"};
  v_fut = verify_inner_product(transcript, drv, descriptor_p, product, a_commit, l_vector, r_vector,
                               ap_value);
  xens::get_scheduler().run();
  if (v_fut.value()) {
    baser::panic("verification should fail");
  }
}
