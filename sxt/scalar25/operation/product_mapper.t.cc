#include "sxt/scalar25/operation/product_mapper.h"

#include <cstddef>

#include "sxt/base/device/memory_utility.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/base/stream.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/device_resource.h"
#include "sxt/scalar25/operation/overload.h"
#include "sxt/scalar25/type/literal.h"

using namespace sxt;
using namespace sxt::s25o;
using sxt::s25t::operator""_s25;

TEST_CASE("we can map two arrays of scalars to their product") {
  SECTION("map_index gives us the product of two scalars") {
    memmg::managed_array<s25t::element> a = {0x123_s25};
    memmg::managed_array<s25t::element> b = {0x321_s25};
    product_mapper mapper{a.data(), b.data()};
    REQUIRE(mapper.map_index(0) == a[0] * b[0]);
    s25t::element res;
    mapper.map_index(res, 0);
    REQUIRE(res == a[0] * b[0]);
  }

  SECTION("we can convert a mapper of device memory to a mapper of host memory") {
    memmg::managed_array<s25t::element> a = {0x1_s25, 0x2_s25};
    memmg::managed_array<s25t::element> b = {0x3_s25, 0x4_s25};
    memmg::managed_array<s25t::element> a_dev{2, memr::get_device_resource()};
    memmg::managed_array<s25t::element> b_dev{2, memr::get_device_resource()};
    basdv::memcpy_host_to_device(a_dev.data(), a.data(), a.size() * sizeof(s25t::element));
    basdv::memcpy_host_to_device(b_dev.data(), b.data(), b.size() * sizeof(s25t::element));

    memmg::managed_array<std::byte> c_host(product_mapper::num_bytes_per_index);
    xenb::stream stream;
    product_mapper mapper{a_dev.data(), b_dev.data()};
    auto mapper_p = mapper.async_make_host_mapper(c_host.data(), stream, 2, 1);
    basdv::synchronize_stream(stream);
    REQUIRE(mapper_p.map_index(0) == 0x8_s25);
  }
}
