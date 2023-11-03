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
#include "sxt/multiexp/bucket_method/accumulation_kernel.h"

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxbk;

TEST_CASE("we can accumulate the buckets for a multi-exponentiation") {
  using E = bascrv::element97;
  memmg::managed_array<E> bucket_sums(255, memr::get_managed_device_resource());
  memmg::managed_array<uint8_t> scalars(memr::get_managed_device_resource());
  memmg::managed_array<E> generators(memr::get_managed_device_resource());

  SECTION("we handle the empty case") {
    bucket_accumulate<<<1, 1>>>(bucket_sums.data(), generators.data(), scalars.data(), 0);
    basdv::synchronize_device();
    for (unsigned i = 0; i < bucket_sums.size(); ++i) {
      REQUIRE(bucket_sums[i] == E::identity());
    }
  }

  SECTION("we handle an accumulation with a single element") {
    scalars = {2};
    generators = {123};
    bucket_accumulate<<<1, 1>>>(bucket_sums.data(), generators.data(), scalars.data(), 1);
    basdv::synchronize_device();
    for (unsigned i = 0; i < bucket_sums.size(); ++i) {
      if (i + 1 != scalars[0]) {
        REQUIRE(bucket_sums[i] == E::identity());
      } else {
        REQUIRE(bucket_sums[i] == generators[0]);
      }
    }
  }

  SECTION("we ignore zero scalars") {
    scalars = {0};
    generators = {123};
    bucket_accumulate<<<1, 1>>>(bucket_sums.data(), generators.data(), scalars.data(), 1);
    basdv::synchronize_device();
    for (unsigned i = 0; i < bucket_sums.size(); ++i) {
      REQUIRE(bucket_sums[i] == E::identity());
    }
  }

  SECTION("we can accumulate multiple elements") {
    scalars = {2, 2};
    generators = {123, 456};
    bucket_accumulate<<<1, 1>>>(bucket_sums.data(), generators.data(), scalars.data(), 2);
    basdv::synchronize_device();
    for (unsigned i = 0; i < bucket_sums.size(); ++i) {
      if (i == 1) {
        REQUIRE(bucket_sums[i] == (123 + 456));
      } else {
        REQUIRE(bucket_sums[i] == E::identity());
      }
    }
  }

  SECTION("we can do an accumulation with multiple blocks") {
    bucket_sums.resize(255 * 2);
    scalars = {2, 2};
    generators = {123, 456};
    bucket_accumulate<<<2, 1>>>(bucket_sums.data(), generators.data(), scalars.data(), 2);
    basdv::synchronize_device();
    for (unsigned i = 0; i < bucket_sums.size(); ++i) {
      if (i == 1) {
        REQUIRE(bucket_sums[i] == 123);
      } else if (i == 255 + 1) {
        REQUIRE(bucket_sums[i] == 456);
      } else {
        REQUIRE(bucket_sums[i] == E::identity());
      }
    }
  }

  SECTION("we can do an accumulation with multiple outputs") {
    bucket_sums.resize(255 * 2);
    scalars = {2, 3};
    generators = {123};
    bucket_accumulate<<<dim3(1, 2, 1), 1>>>(bucket_sums.data(), generators.data(), scalars.data(),
                                            1);
    basdv::synchronize_device();
    for (unsigned i = 0; i < bucket_sums.size(); ++i) {
      if (i == 1) {
        REQUIRE(bucket_sums[i] == 123);
      } else if (i == 255 + 2) {
        REQUIRE(bucket_sums[i] == 123);
      } else {
        REQUIRE(bucket_sums[i] == E::identity());
      }
    }
  }

  SECTION("we can accumulate with multiple bytes") {
    bucket_sums.resize(255 * 2);
    scalars = {2, 3};
    generators = {123};
    bucket_accumulate<<<1, 2>>>(bucket_sums.data(), generators.data(), scalars.data(), 1);
    basdv::synchronize_device();
    for (unsigned i = 0; i < bucket_sums.size(); ++i) {
      if (i == 1) {
        REQUIRE(bucket_sums[i] == 123);
      } else if (i == 255 + 2) {
        REQUIRE(bucket_sums[i] == 123);
      } else {
        REQUIRE(bucket_sums[i] == E::identity());
      }
    }
  }
}
