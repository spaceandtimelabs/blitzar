#include "sxt/execution/base/stream.h"

#include "sxt/base/device/property.h"
#include "sxt/base/test/unit_test.h"
#include "sxt/execution/base/stream_pool.h"

using namespace sxt;
using namespace sxt::xenb;

TEST_CASE("stream provides a wrapper around pooled CUDA streams") {
  if (basdv::get_num_devices() == 0) {
    return;
  }

  SECTION("default construction gives us a non-null stream") {
    stream s;
    REQUIRE(s.raw_stream() != nullptr);
  }

  SECTION("we can release a stream") {
    stream s1;
    auto ptr = s1.release_handle();
    REQUIRE(s1.release_handle() == nullptr);
    REQUIRE(ptr != nullptr);
    get_stream_pool()->release_handle(ptr);
  }

  SECTION("we can move construct streams") {
    stream s1;
    auto ptr = s1.raw_stream();
    stream s2{std::move(s1)};
    REQUIRE(s1.release_handle() == nullptr);
    REQUIRE(s2.raw_stream() == ptr);
  }

  SECTION("we can move assign streams") {
    stream s1;
    auto ptr = s1.raw_stream();
    stream s2;
    s2 = std::move(s1);
    REQUIRE(s1.release_handle() == nullptr);
    REQUIRE(s2.raw_stream() == ptr);
  }
}
