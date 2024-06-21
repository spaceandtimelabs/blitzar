/** Proofs GPU - Space and Time's cryptographic proof algorithms on the CPU and GPU.
 *
 * Copyright 2023-present Space and Time Labs, Inc.
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
/**
 * Adopted from zkcrypto/bls12_381
 *
 * Copyright (c) 2021
 * Sean Bowe <ewillbefull@gmail.com>
 * Jack Grigg <thestr4d@gmail.com>
 *
 * See third_party/license/zkcrypto.LICENSE
 */
#include "sxt/curve_gkg1/type/element_affine.h"

#include "sxt/base/test/unit_test.h"
#include "sxt/fieldgk/constant/one.h"
#include "sxt/fieldgk/constant/zero.h"
#include "sxt/fieldgk/type/literal.h"

using namespace sxt;
using namespace sxt::ck1t;
using fgkt::operator""_fgk;

TEST_CASE("affine element equality") {
  SECTION("can distinguish the generator from the identity") {
    element_affine generator{fgkcn::one_v,
                             0x14b34cf69dc25d68aa7b8cf435dfafbb23d3446f21c77dc311b2dff1448c41d8_fgk,
                             false};

    constexpr element_affine identity{fgkcn::zero_v, fgkcn::one_v, true};

    REQUIRE(generator == generator);
    REQUIRE(identity == identity);
    REQUIRE(generator != identity);
    REQUIRE(identity != generator);
  }
}
