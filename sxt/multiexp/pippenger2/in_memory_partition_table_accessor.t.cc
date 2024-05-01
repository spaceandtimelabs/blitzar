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
#include "sxt/multiexp/pippenger2/in_memory_partition_table_accessor.h"

#include <vector>

#include "sxt/base/curve/example_element.h"
#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/temp_file.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"

using namespace sxt;
using namespace sxt::mtxpp2;

TEST_CASE("we can provide access to precomputed partition sums stored on disk") {
  bastst::temp_file temp_file{std::ios::binary};

  using E = bascrv::element97;
  basdv::stream stream;

  SECTION("we can access a single element") {
    E e{11};
    temp_file.stream().write(reinterpret_cast<const char*>(&e), sizeof(e));
    temp_file.stream().close();
    in_memory_partition_table_accessor<E> accessor{temp_file.name()};
    memmg::managed_array<E> v_dev{1, memr::get_device_resource()};
    accessor.async_copy_precomputed_sums_to_device(v_dev, stream, 0);
    std::vector<E> v(1);
    basdv::async_copy_device_to_host(v, v_dev, stream);
    basdv::synchronize_stream(stream);
    std::vector<E> expected = {e};
    REQUIRE(v == expected);
  }

  SECTION("we can access a elements with offset") {
    std::vector<E> data((1u << 16u) * 2);
    data[1u << 16u] = 12u;
    temp_file.stream().write(reinterpret_cast<const char*>(data.data()), sizeof(E) * data.size());
    temp_file.stream().close();
    in_memory_partition_table_accessor<E> accessor{temp_file.name()};
    memmg::managed_array<E> v_dev{1, memr::get_device_resource()};
    accessor.async_copy_precomputed_sums_to_device(v_dev, stream, 1);
    std::vector<E> v(1);
    basdv::async_copy_device_to_host(v, v_dev, stream);
    basdv::synchronize_stream(stream);
    std::vector<E> expected = {12u};
    REQUIRE(v == expected);
  }

  SECTION("we can write an accessor to a file") {
    memmg::managed_array<E> data((1u << 16u) * 2);
    unsigned cnt = 0;
    for (auto& val : data) {
      val = cnt++;
    }
    in_memory_partition_table_accessor<E> accessor{memmg::managed_array<E>{data}};
    temp_file.stream().close();
    accessor.write_to_file(temp_file.name());
    in_memory_partition_table_accessor<E> accessor_p{temp_file.name()};
    memmg::managed_array<E> data_dev{data.size(), memr::get_device_resource()};
    accessor_p.async_copy_precomputed_sums_to_device(data_dev, stream, 0);
    memmg::managed_array<E> data_p(data.size());
    basdv::async_copy_device_to_host(data_p, data_dev, stream);
    basdv::synchronize_stream(stream);
    REQUIRE(data == data_p);
  }
}
