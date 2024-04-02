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
    std::vector<E> data{11, 12};
    temp_file.stream().write(reinterpret_cast<const char*>(data.data()), sizeof(E) * data.size());
    temp_file.stream().close();
    in_memory_partition_table_accessor<E> accessor{temp_file.name()};
    memmg::managed_array<E> v_dev{1, memr::get_device_resource()};
    accessor.async_copy_precomputed_sums_to_device(v_dev, stream, 1);
    std::vector<E> v(1);
    basdv::async_copy_device_to_host(v, v_dev, stream);
    basdv::synchronize_stream(stream);
    std::vector<E> expected = {data[1]};
    REQUIRE(v == expected);
  }
}
