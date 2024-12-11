#include "sxt/base/iterator/split.h"

#include <iterator>

#include "sxt/base/test/unit_test.h"
using namespace sxt;
using namespace sxt::basit;

TEST_CASE("we can split an index_range") {
  split_options options;

  SECTION("we handle the n=1 case") {
    auto [first, last] = split(index_range{1, 9}, options);
    REQUIRE(std::distance(first, last) == 1);
    REQUIRE(*first == index_range{1, 9});
  }

  SECTION("we can split an empty range") {
    auto [first, last] = split(index_range{3, 3}, options);
    REQUIRE(std::distance(first, last) == 0);
  }

  SECTION("we handle the case when n is greater than 1") {
    options.split_factor = 2;
    auto [iter, last] = split(index_range{1, 5}, options);
    REQUIRE(std::distance(iter, last) == 2);
    REQUIRE(*iter++ == index_range{1, 3});
    REQUIRE(*iter++ == index_range{3, 5});
    REQUIRE(iter == last);
  }

  SECTION("we handle the case when n doesn't divide the range size") {
    options.split_factor = 3;
    auto [iter, last] = split(index_range{1, 5}, options);
    REQUIRE(std::distance(iter, last) == 2);
    REQUIRE(*iter++ == index_range{1, 3});
    REQUIRE(*iter++ == index_range{3, 5});
    REQUIRE(iter == last);
  }

  SECTION("we handle the case when n is greater than the range") {
    options.split_factor = 10;
    auto [iter, last] = split(index_range{1, 3}, options);
    REQUIRE(std::distance(iter, last) == 2);
    REQUIRE(*iter++ == index_range{1, 2});
    REQUIRE(*iter++ == index_range{2, 3});
    REQUIRE(iter == last);
  }

  SECTION("we respect the min chunk size") {
    options.min_chunk_size = 2;
    options.split_factor = 4;
    auto [iter, last] = split(index_range{0, 4}, options);
    REQUIRE(std::distance(iter, last) == 2);
    REQUIRE(*iter++ == index_range{0, 2});
    REQUIRE(*iter++ == index_range{2, 4});
    REQUIRE(iter == last);
  }

  SECTION("we respect the max chunk size") {
    options.max_chunk_size = 2;
    auto [iter, last] = split(index_range{0, 4}, options);
    REQUIRE(std::distance(iter, last) == 2);
    REQUIRE(*iter++ == index_range{0, 2});
    REQUIRE(*iter++ == index_range{2, 4});
    REQUIRE(iter == last);
  }

  SECTION("we respect the chunk multiple") {
    options.split_factor = 4;
    auto [iter, last] = split(index_range{0, 4}.chunk_multiple(3), options);
    REQUIRE(std::distance(iter, last) == 2);
    REQUIRE(*iter++ == index_range{0, 3});
    REQUIRE(*iter++ == index_range{3, 4});
    REQUIRE(iter == last);
  }
}
