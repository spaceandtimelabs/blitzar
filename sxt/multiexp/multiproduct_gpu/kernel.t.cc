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
#include "sxt/multiexp/multiproduct_gpu/kernel.h"

#include <numeric>

#include "sxt/algorithm/base/gather_mapper.h"
#include "sxt/algorithm/reduction/test_reducer.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"

using namespace sxt;
using namespace sxt::mtxmpg;

TEST_CASE("we can compute multiproducts using the GPU") {
  memmg::managed_array<uint64_t> generators(100, memr::get_managed_device_resource());
  std::iota(generators.begin(), generators.end(), 1);

  using Mapper = algb::gather_mapper<uint64_t>;

  SECTION("we can compute a single multiproduct with a single term") {
    memmg::managed_array<unsigned> indexes(1, memr::get_managed_device_resource());
    indexes = {10};
    memmg::managed_array<block_computation_descriptor> block_descriptors{
        memr::get_managed_device_resource()};
    block_descriptors = {
        {
            .block_offset = 0,
            .index_first = 0,
            .n = 1,
            .reduction_num_blocks = 1,
            .block_size = xenk::block_size_t::v1,
        },
    };
    memmg::managed_array<uint64_t> res(block_descriptors.size(),
                                       memr::get_managed_device_resource());
    multiproduct_kernel<algr::test_add_reducer, Mapper><<<1, 1, sizeof(uint64_t) * 1>>>(
        res.data(), generators.data(), indexes.data(), block_descriptors.data());
    basdv::synchronize_device();
    memmg::managed_array<uint64_t> expected = {11};
    REQUIRE(res == expected);
  }

  SECTION("we can compute multiple multiproducts with varying number of terms") {
    memmg::managed_array<unsigned> indexes(1, memr::get_managed_device_resource());
    indexes = {10, 13, 15, 19, 21};
    memmg::managed_array<block_computation_descriptor> block_descriptors{
        memr::get_managed_device_resource()};
    block_descriptors = {
        {
            .block_offset = 0,
            .index_first = 0,
            .n = 1,
            .reduction_num_blocks = 1,
            .block_size = xenk::block_size_t::v1,
        },
        {
            .block_offset = 1,
            .index_first = 1,
            .n = 4,
            .reduction_num_blocks = 1,
            .block_size = xenk::block_size_t::v2,
        },
    };
    memmg::managed_array<uint64_t> res(block_descriptors.size(),
                                       memr::get_managed_device_resource());
    multiproduct_kernel<algr::test_add_reducer, Mapper><<<2, 2, sizeof(uint64_t) * 2>>>(
        res.data(), generators.data(), indexes.data(), block_descriptors.data());
    basdv::synchronize_device();
    memmg::managed_array<uint64_t> expected = {
        11,
        14 + 16 + 20 + 22,
    };
    REQUIRE(res == expected);
  }

  SECTION("we can compute a multiproduct with more than a single block") {
    memmg::managed_array<unsigned> indexes(1, memr::get_managed_device_resource());
    indexes = {1, 3, 5, 7, 14};
    memmg::managed_array<block_computation_descriptor> block_descriptors{
        memr::get_managed_device_resource()};
    block_descriptors = {
        {
            .block_offset = 0,
            .index_first = 0,
            .n = 5,
            .reduction_num_blocks = 2,
            .block_size = xenk::block_size_t::v1,
        },
        {
            .block_offset = 0,
            .index_first = 0,
            .n = 5,
            .reduction_num_blocks = 2,
            .block_size = xenk::block_size_t::v1,
        },
    };
    memmg::managed_array<uint64_t> res(block_descriptors.size(),
                                       memr::get_managed_device_resource());
    multiproduct_kernel<algr::test_add_reducer, Mapper><<<2, 1, sizeof(uint64_t) * 1>>>(
        res.data(), generators.data(), indexes.data(), block_descriptors.data());
    basdv::synchronize_device();
    memmg::managed_array<uint64_t> expected = {
        2 + 4 + 15,
        6 + 8,
    };
    REQUIRE(res == expected);
  }
}
