#include "sxt/algorithm/transform/prefix_sum.h"

#include "sxt/base/device/stream.h"
#include "sxt/base/device/synchronization.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/memory/management/managed_array.h"
#include "sxt/memory/resource/managed_device_resource.h"
using namespace sxt;
using namespace sxt::algtr;

TEST_CASE("we can compute exclusive prefix sums") {
  basdv::stream stream;

  memmg::managed_array<unsigned> in{memr::get_managed_device_resource()};
  memmg::managed_array<unsigned> out{memr::get_managed_device_resource()};

  SECTION("we handle the empty case") {
    sxt::algtr::exclusive_prefix_sum(out, in, stream);
  }

  SECTION("we handle the case of a single element") {
    in = {123};
    out.resize(1);
    sxt::algtr::exclusive_prefix_sum(out, in, stream);
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected = {0};
    REQUIRE(out == expected);
  }

  SECTION("we handle two elements") {
    in = {123, 456};
    out.resize(2);
    sxt::algtr::exclusive_prefix_sum(out, in, stream);
    basdv::synchronize_stream(stream);
    memmg::managed_array<unsigned> expected = {0, 123};
    REQUIRE(out == expected);
  }
}
