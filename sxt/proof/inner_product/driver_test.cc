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
#include "sxt/proof/inner_product/driver_test.h"

#include <memory_resource>

#include "sxt/base/num/fast_random_number_generator.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/curve21/operation/overload.h"
#include "sxt/execution/async/future.h"
#include "sxt/execution/schedule/scheduler.h"
#include "sxt/proof/inner_product/driver.h"
#include "sxt/proof/inner_product/proof_descriptor.h"
#include "sxt/proof/inner_product/random_product_generation.h"
#include "sxt/proof/inner_product/workspace.h"
#include "sxt/ristretto/operation/compression.h"
#include "sxt/ristretto/operation/overload.h"
#include "sxt/ristretto/random/element.h"
#include "sxt/ristretto/type/compressed_element.h"
#include "sxt/scalar25/operation/inv.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/random/element.h"

namespace sxt::prfip {
//--------------------------------------------------------------------------------------------------
// exercise_fold_commit
//--------------------------------------------------------------------------------------------------
static void exercise_fold_commit(const driver& drv) {
  std::pmr::monotonic_buffer_resource alloc;
  proof_descriptor descriptor;
  basct::cspan<s25t::element> a_vector;
  basn::fast_random_number_generator rng{1, 2};

  rstt::compressed_element l_value, r_value;
  rstt::compressed_element expected_l_value, expected_r_value;
  c21t::element_p3 t;

  auto& g_vector = descriptor.g_vector;
  auto& b_vector = descriptor.b_vector;
  auto& q_value = descriptor.q_value;

  SECTION("we correctly commit for the n = 2 case") {
    generate_random_product(descriptor, a_vector, rng, &alloc, 2);

    auto workspace_fut = drv.make_workspace(descriptor, a_vector);
    xens::get_scheduler().run();
    auto workspace = std::move(workspace_fut.value());

    auto fut = drv.commit_to_fold(l_value, r_value, *workspace);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    // l_value
    t = a_vector[0] * g_vector[1] + (a_vector[0] * b_vector[1]) * *q_value;
    rsto::compress(expected_l_value, t);
    REQUIRE(l_value == expected_l_value);

    // r_value
    t = a_vector[1] * g_vector[0] + (a_vector[1] * b_vector[0]) * *q_value;
    rsto::compress(expected_r_value, t);
    REQUIRE(r_value == expected_r_value);
  }

  SECTION("we correctly commit for the n = 3 case") {
    generate_random_product(descriptor, a_vector, rng, &alloc, 3);

    auto workspace_fut = drv.make_workspace(descriptor, a_vector);
    xens::get_scheduler().run();
    auto workspace = std::move(workspace_fut.value());
    auto fut = drv.commit_to_fold(l_value, r_value, *workspace);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    // l_value
    t = a_vector[0] * g_vector[2] + a_vector[1] * g_vector[3] +
        (a_vector[0] * b_vector[2]) * *q_value;
    rsto::compress(expected_l_value, t);
    REQUIRE(l_value == expected_l_value);

    // r_value
    t = a_vector[2] * g_vector[0] + (a_vector[2] * b_vector[0]) * *q_value;
    rsto::compress(expected_r_value, t);
    REQUIRE(r_value == expected_r_value);
  }

  SECTION("we correctly commit for the n = 4 case") {
    generate_random_product(descriptor, a_vector, rng, &alloc, 4);
    auto workspace_fut = drv.make_workspace(descriptor, a_vector);
    xens::get_scheduler().run();
    auto workspace = std::move(workspace_fut.value());
    auto fut = drv.commit_to_fold(l_value, r_value, *workspace);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    // l_value
    t = a_vector[0] * g_vector[2] + a_vector[1] * g_vector[3] +
        (a_vector[0] * b_vector[2] + a_vector[1] * b_vector[3]) * *q_value;
    rsto::compress(expected_l_value, t);
    REQUIRE(l_value == expected_l_value);

    // r_value
    t = a_vector[2] * g_vector[0] + a_vector[3] * g_vector[1] +
        (a_vector[2] * b_vector[0] + a_vector[3] * b_vector[1]) * *q_value;
    rsto::compress(expected_r_value, t);
    REQUIRE(r_value == expected_r_value);
  }
}

//--------------------------------------------------------------------------------------------------
// exercise_fold
//--------------------------------------------------------------------------------------------------
static void exercise_fold(const driver& drv) {
  std::pmr::monotonic_buffer_resource alloc;
  proof_descriptor descriptor;
  basct::cspan<s25t::element> a_vector;
  basn::fast_random_number_generator rng{1, 2};

  rstt::compressed_element l_value, r_value;
  rstt::compressed_element expected_l_value, expected_r_value;
  c21t::element_p3 t;

  auto& g_vector = descriptor.g_vector;
  auto& b_vector = descriptor.b_vector;
  auto& q_value = descriptor.q_value;

  s25t::element ap_value;

  s25t::element x1, x1_inv, x2, x2_inv;
  s25rn::generate_random_element(x1, rng);
  s25o::inv(x1_inv, x1);
  s25rn::generate_random_element(x2, rng);
  s25o::inv(x2_inv, x2);

  SECTION("we handle the n = 2 case") {
    generate_random_product(descriptor, a_vector, rng, &alloc, 2);

    auto workspace_fut = drv.make_workspace(descriptor, a_vector);
    xens::get_scheduler().run();
    auto workspace = std::move(workspace_fut.value());

    auto fut = drv.fold(*workspace, x1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    auto ap_fut = workspace->ap_value(ap_value);
    xens::get_scheduler().run();
    REQUIRE(ap_fut.ready());

    auto expected_ap_value = x1 * a_vector[0] + x1_inv * a_vector[1];
    REQUIRE(ap_value == expected_ap_value);
  }

  SECTION("we handle the n = 3 case") {
    generate_random_product(descriptor, a_vector, rng, &alloc, 3);

    auto workspace_fut = drv.make_workspace(descriptor, a_vector);
    xens::get_scheduler().run();
    auto workspace = std::move(workspace_fut.value());

    auto fut = drv.fold(*workspace, x1);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    fut = drv.commit_to_fold(l_value, r_value, *workspace);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    auto a1 = x1 * a_vector[0] + x1_inv * a_vector[2];
    auto a2 = x1 * a_vector[1];

    auto b1 = x1_inv * b_vector[0] + x1 * b_vector[2];
    auto b2 = x1_inv * b_vector[1];

    auto g1 = x1_inv * g_vector[0] + x1 * g_vector[2];
    auto g2 = x1_inv * g_vector[1] + x1 * g_vector[3];

    // l_value
    t = a1 * g2 + (a1 * b2) * *q_value;
    rsto::compress(expected_l_value, t);
    REQUIRE(l_value == expected_l_value);

    // r_value
    t = a2 * g1 + (a2 * b1) * *q_value;
    rsto::compress(expected_r_value, t);
    REQUIRE(r_value == expected_r_value);

    // ap_value
    fut = drv.fold(*workspace, x2);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    auto ap_fut = workspace->ap_value(ap_value);
    xens::get_scheduler().run();
    REQUIRE(ap_fut.ready());
    auto expected_ap_value = x2 * a1 + x2_inv * a2;
    REQUIRE(ap_value == expected_ap_value);
  }
}

//--------------------------------------------------------------------------------------------------
// exercise_verification
//--------------------------------------------------------------------------------------------------
static void exercise_verification(const driver& drv) {
  std::pmr::monotonic_buffer_resource alloc;
  proof_descriptor descriptor;
  basct::cspan<s25t::element> a_vector;
  basn::fast_random_number_generator rng{1, 2};

  rstt::compressed_element commit;

  auto& g_vector = descriptor.g_vector;
  auto& b_vector = descriptor.b_vector;
  auto& q_value = descriptor.q_value;

  SECTION("we handle the n = 1 case") {
    generate_random_product(descriptor, a_vector, rng, &alloc, 1);
    std::vector<rstt::compressed_element> l_vector(0), r_vector(0);
    std::vector<s25t::element> x_vector(0);
    auto fut = drv.compute_expected_commitment(commit, descriptor, l_vector, r_vector, x_vector,
                                               a_vector[0]);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    rstt::compressed_element expected;
    rsto::compress(expected, a_vector[0] * b_vector[0] * *q_value + a_vector[0] * g_vector[0]);
    REQUIRE(commit == expected);
  }

  SECTION("we handle the n = 2 case") {
    generate_random_product(descriptor, a_vector, rng, &alloc, 2);

    std::vector<rstt::compressed_element> l_vector(1), r_vector(1);
    rstrn::generate_random_elements(l_vector, rng);
    rstrn::generate_random_elements(r_vector, rng);

    std::vector<s25t::element> x_vector(1), x_inv_vector(1);
    s25rn::generate_random_elements(x_vector, rng);
    s25o::batch_inv(x_inv_vector, x_vector);

    auto folded_a = x_vector[0] * a_vector[0] + x_inv_vector[0] * a_vector[1];

    auto fut =
        drv.compute_expected_commitment(commit, descriptor, l_vector, r_vector, x_vector, folded_a);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    auto folded_b = x_inv_vector[0] * b_vector[0] + x_vector[0] * b_vector[1];

    auto folded_g = x_inv_vector[0] * g_vector[0] + x_vector[0] * g_vector[1];

    auto expected = folded_a * folded_b * *q_value + folded_a * folded_g -
                    x_vector[0] * x_vector[0] * l_vector[0] -
                    x_inv_vector[0] * x_inv_vector[0] * r_vector[0];
    REQUIRE(commit == expected);
  }

  SECTION("we handle the n = 3 case") {
    generate_random_product(descriptor, a_vector, rng, &alloc, 3);

    std::vector<rstt::compressed_element> l_vector(2), r_vector(2);
    rstrn::generate_random_elements(l_vector, rng);
    rstrn::generate_random_elements(r_vector, rng);

    std::vector<s25t::element> x_vector(2), x_inv_vector(2);
    s25rn::generate_random_elements(x_vector, rng);
    s25o::batch_inv(x_inv_vector, x_vector);

    auto folded_a = x_vector[0] * x_vector[1] * a_vector[0] +
                    x_vector[0] * x_inv_vector[1] * a_vector[1] +
                    x_inv_vector[0] * x_vector[1] * a_vector[2];

    auto fut =
        drv.compute_expected_commitment(commit, descriptor, l_vector, r_vector, x_vector, folded_a);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    auto folded_b = x_inv_vector[0] * x_inv_vector[1] * b_vector[0] +
                    x_inv_vector[0] * x_vector[1] * b_vector[1] +
                    x_vector[0] * x_inv_vector[1] * b_vector[2];

    auto folded_g = x_inv_vector[0] * x_inv_vector[1] * g_vector[0] +
                    x_inv_vector[0] * x_vector[1] * g_vector[1] +
                    x_vector[0] * x_inv_vector[1] * g_vector[2] +
                    x_vector[0] * x_vector[1] * g_vector[3];

    auto expected = folded_a * folded_b * *q_value + folded_a * folded_g -
                    x_vector[0] * x_vector[0] * l_vector[0] -
                    x_vector[1] * x_vector[1] * l_vector[1] -
                    x_inv_vector[0] * x_inv_vector[0] * r_vector[0] -
                    x_inv_vector[1] * x_inv_vector[1] * r_vector[1];
    REQUIRE(commit == expected);
  }

  SECTION("we handle the n = 4 case") {
    generate_random_product(descriptor, a_vector, rng, &alloc, 4);

    std::vector<rstt::compressed_element> l_vector(2), r_vector(2);
    rstrn::generate_random_elements(l_vector, rng);
    rstrn::generate_random_elements(r_vector, rng);

    std::vector<s25t::element> x_vector(2), x_inv_vector(2);
    s25rn::generate_random_elements(x_vector, rng);
    s25o::batch_inv(x_inv_vector, x_vector);

    auto folded_a = x_vector[0] * x_vector[1] * a_vector[0] +
                    x_vector[0] * x_inv_vector[1] * a_vector[1] +
                    x_inv_vector[0] * x_vector[1] * a_vector[2] +
                    x_inv_vector[0] * x_inv_vector[1] * a_vector[3];

    auto fut =
        drv.compute_expected_commitment(commit, descriptor, l_vector, r_vector, x_vector, folded_a);
    xens::get_scheduler().run();
    REQUIRE(fut.ready());

    auto folded_b = x_inv_vector[0] * x_inv_vector[1] * b_vector[0] +
                    x_inv_vector[0] * x_vector[1] * b_vector[1] +
                    x_vector[0] * x_inv_vector[1] * b_vector[2] +
                    x_vector[0] * x_vector[1] * b_vector[3];

    auto folded_g = x_inv_vector[0] * x_inv_vector[1] * g_vector[0] +
                    x_inv_vector[0] * x_vector[1] * g_vector[1] +
                    x_vector[0] * x_inv_vector[1] * g_vector[2] +
                    x_vector[0] * x_vector[1] * g_vector[3];

    auto expected = folded_a * folded_b * *q_value + folded_a * folded_g -
                    x_vector[0] * x_vector[0] * l_vector[0] -
                    x_vector[1] * x_vector[1] * l_vector[1] -
                    x_inv_vector[0] * x_inv_vector[0] * r_vector[0] -
                    x_inv_vector[1] * x_inv_vector[1] * r_vector[1];
    REQUIRE(commit == expected);
  }
}

//--------------------------------------------------------------------------------------------------
// exercise_driver
//--------------------------------------------------------------------------------------------------
void exercise_driver(const driver& drv) {
  SECTION("we can commit to the fold of an inner product problem") { exercise_fold_commit(drv); }

  SECTION("we can fold an inner product problem") { exercise_fold(drv); }

  SECTION("we can compute the expected commitment needed for verification") {
    exercise_verification(drv);
  }
}
} // namespace sxt::prfip
