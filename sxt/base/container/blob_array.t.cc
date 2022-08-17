#include "sxt/base/container/blob_array.h"

#include <iterator>

#include "sxt/base/test/unit_test.h"

using namespace sxt::basct;

TEST_CASE("we can manage an array of blobs") {
  blob_array arr;

  SECTION("an array starts empty") {
    REQUIRE(arr.empty());
    REQUIRE(arr.size() == 0);
  }

  SECTION("we can construct an array with elements") {
    arr = blob_array{10, 2};
    REQUIRE(arr.size() == 10);
    REQUIRE(arr.blob_size() == 2);
    REQUIRE(arr[0].size() == 2);
    REQUIRE(std::distance(arr[0].data(), arr[1].data()) == 2);
  }
}
